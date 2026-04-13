# Opening Penalty Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add instrumentation to capture raw priors, post-root-adjustment priors, and visit distributions at each opening root search, enabling diagnosis of why edge/corner play regresses.

**Architecture:** Per-game diagnostic records are built in the self-play game loop, passed through IPC to the trainer, written to per-game JSON files, and aggregated into the per-iteration stats sidecar. A single shared helper classifies moves into regions, ensuring zero drift between raw records and aggregates.

**Tech Stack:** Python, numpy (for percentile computation in aggregation), existing MCTS/self-play/trainer infrastructure

**Spec:** `docs/superpowers/specs/2026-04-12-opening-penalty-diagnostics-design.md`

---

## File Structure

| File | Responsibility | Change Type |
|------|---------------|-------------|
| `scripts/GPU/alphazero/opening_diagnostics.py` | **NEW** — Shared helper: regional classification, per-root record builder, sidecar aggregation | Create |
| `scripts/GPU/alphazero/self_play.py` | Build per-root diagnostic records in game loop, attach to GameRecord | Modify |
| `scripts/GPU/alphazero/ipc_messages.py` | Add `opening_diagnostics` field to GameComplete | Modify |
| `scripts/GPU/alphazero/self_play_worker.py` | Pass diagnostics from GameRecord to GameComplete message | Modify |
| `scripts/GPU/alphazero/game_saver.py` | Accept and write diagnostics + meta to per-game JSON | Modify |
| `scripts/GPU/alphazero/trainer.py` | Collect per-game diagnostics, call aggregation, write to sidecar | Modify |

---

### Task 1: Create the shared opening_diagnostics module

This is the single source of truth for regional classification and record building. Both per-game records and sidecar aggregation use the same functions — zero drift by construction.

**Files:**
- Create: `scripts/GPU/alphazero/opening_diagnostics.py`

- [ ] **Step 1: Create the module with regional classification helpers**

```python
"""Opening penalty diagnostics — shared helpers.

This module is the single source of truth for:
- Regional classification of moves (edge_band, near_corner, interior)
- Per-root diagnostic record building
- Sidecar aggregation of per-game records

Both per-game records and sidecar aggregates use the same classification
functions, ensuring zero drift between raw records and summaries.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


# --- Regional classification (must match mcts.py penalty definitions) ---

def is_edge_band(r: int, c: int, board_size: int, band_width: int) -> bool:
    """True if (r, c) is in the edge band. Same definition as mcts._is_edge_band."""
    return r < band_width or r >= board_size - band_width or c < band_width or c >= board_size - band_width


def is_near_corner(r: int, c: int, board_size: int, radius: int) -> bool:
    """True if (r, c) is within Chebyshev distance <= radius of any corner.
    Same definition as mcts._is_near_corner_cheb.
    """
    if radius <= 0:
        return False
    corners = ((0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1))
    for rr, cc in corners:
        if max(abs(r - rr), abs(c - cc)) <= radius:
            return True
    return False


def primary_region(r: int, c: int, board_size: int, band_width: int, corner_radius: int) -> str:
    """Exclusive region assignment: near_corner > edge_band > interior."""
    if is_near_corner(r, c, board_size, corner_radius):
        return "near_corner"
    if is_edge_band(r, c, board_size, band_width):
        return "edge_band"
    return "interior"


def classify_move(r: int, c: int, board_size: int, band_width: int, corner_radius: int) -> dict:
    """Full classification with boolean flags + primary region."""
    eb = is_edge_band(r, c, board_size, band_width)
    nc = is_near_corner(r, c, board_size, corner_radius)
    pri = "near_corner" if nc else ("edge_band" if eb else "interior")
    return {"is_edge_band": eb, "is_near_corner": nc, "primary_region": pri}


# --- Diagnostic window computation ---

def compute_diagnostic_end_ply(edge_penalty_ply: int, corner_penalty_ply: int, floor: int = 4, extra: int = 2) -> Tuple[int, bool]:
    """Compute the diagnostic end ply and whether the floor was used.

    Returns:
        (diagnostic_end_ply, used_floor)
    """
    max_penalty_ply = max(edge_penalty_ply, corner_penalty_ply)
    used_floor = max_penalty_ply < floor
    return max(max_penalty_ply, floor) + extra, used_floor


# --- Per-root record builder ---

def _normalize_priors_to_rc(priors: Dict, decode_fn) -> Dict[Tuple[int, int], float]:
    """Normalize prior dict keys to (row, col) tuples.

    Accepts either Dict[int, float] (move_id keys) or Dict[Tuple, float].
    Returns Dict[(row, col), float] in all cases.
    """
    if not priors:
        return {}
    sample_key = next(iter(priors))
    if isinstance(sample_key, tuple):
        return dict(priors)
    # int keys -> decode to (row, col)
    return {decode_fn(mid): p for mid, p in priors.items()}


def build_root_diagnostic(
    ply: int,
    side_to_move: str,
    visit_counts: Dict[Tuple[int, int], int],
    priors_raw: Dict,
    priors_adjusted: Dict,
    board_size: int,
    band_width: int,
    corner_radius: int,
    edge_penalty: float,
    corner_penalty: float,
    edge_penalty_ply: int,
    corner_penalty_ply: int,
    decode_fn,
) -> dict:
    """Build a single per-root diagnostic record.

    All inputs are normalized to (row, col) keys internally.

    Args:
        visit_counts: Dict[(row, col) -> visits] from MCTS
        priors_raw: Dict[move_id or (r,c) -> prior] — raw NN output (root.priors_raw)
        priors_adjusted: Dict[move_id or (r,c) -> prior] — post-root-adjustment
            (includes Dirichlet noise + penalties, not penalty-only)
        decode_fn: Function move_id -> (r, c) for key normalization

    Note: top-1 ties resolve by first-encountered move (iteration order).
    """
    # Normalize all priors to (row, col) keys for uniform processing
    raw_rc = _normalize_priors_to_rc(priors_raw, decode_fn)
    adj_rc = _normalize_priors_to_rc(priors_adjusted, decode_fn)

    edge_active = edge_penalty > 0 and edge_penalty_ply > 0 and ply < edge_penalty_ply
    corner_active = corner_penalty > 0 and corner_penalty_ply > 0 and ply < corner_penalty_ply and corner_radius > 0

    # Build the union of all legal moves from all three sources.
    # visit_counts may omit unvisited moves, so raw/penalized mass would be
    # undercounted if we only iterated visit_counts.
    all_moves: set = set(visit_counts.keys()) | set(raw_rc.keys()) | set(adj_rc.keys())

    # Regional mass accumulators
    regions = ("near_corner", "edge_band", "interior")
    raw_mass = {r: 0.0 for r in regions}
    pen_mass = {r: 0.0 for r in regions}
    visit_mass = {r: 0.0 for r in regions}
    legal_counts = {r: 0 for r in regions}

    total_visits = sum(visit_counts.values())
    total_visits_f = float(max(total_visits, 1))

    # Track top-1 per stage (ties resolve by first encountered move)
    raw_top1_move = None
    raw_top1_share = -1.0
    raw_top1_cls = None
    pen_top1_move = None
    pen_top1_share = -1.0
    pen_top1_cls = None
    visit_top1_move = None
    visit_top1_count = -1
    visit_top1_cls = None

    for (r, c) in all_moves:
        cls = classify_move(r, c, board_size, band_width, corner_radius)
        pri = cls["primary_region"]

        p_raw = raw_rc.get((r, c), 0.0)
        p_adj = adj_rc.get((r, c), 0.0)
        visits = visit_counts.get((r, c), 0)
        v_share = visits / total_visits_f

        legal_counts[pri] += 1
        raw_mass[pri] += p_raw
        pen_mass[pri] += p_adj
        visit_mass[pri] += v_share

        if p_raw > raw_top1_share:
            raw_top1_share = p_raw
            raw_top1_move = [r, c]
            raw_top1_cls = cls
        if p_adj > pen_top1_share:
            pen_top1_share = p_adj
            pen_top1_move = [r, c]
            pen_top1_cls = cls
        if visits > visit_top1_count:
            visit_top1_count = visits
            visit_top1_move = [r, c]
            visit_top1_cls = cls

    def _top1_entry(move, share, cls):
        if move is None:
            return {"move": [0, 0], "share": 0.0, "is_edge_band": False, "is_near_corner": False, "primary_region": "interior"}
        return {
            "move": move,
            "share": round(share, 4),
            "is_edge_band": cls["is_edge_band"],
            "is_near_corner": cls["is_near_corner"],
            "primary_region": cls["primary_region"],
        }

    legal_total = sum(legal_counts.values())

    return {
        "ply": ply,
        "side_to_move": side_to_move,
        "penalties_active": {"edge_band": edge_active, "near_corner": corner_active},
        "config": {
            "edge_band_width": band_width,
            "edge_band_penalty": edge_penalty,
            "near_corner_radius": corner_radius,
            "near_corner_penalty": corner_penalty,
        },
        "legal_moves_total": legal_total,
        "legal_move_counts": {r: legal_counts[r] for r in regions},
        "raw_mass": {r: round(raw_mass[r], 4) for r in regions},
        "penalized_mass": {r: round(pen_mass[r], 4) for r in regions},
        "visit_mass": {r: round(visit_mass[r], 4) for r in regions},
        "raw_top1": _top1_entry(raw_top1_move, raw_top1_share, raw_top1_cls),
        "penalized_top1": _top1_entry(pen_top1_move, pen_top1_share, pen_top1_cls),
        "visit_top1": _top1_entry(visit_top1_move, visit_top1_count / total_visits_f if total_visits > 0 else 0.0, visit_top1_cls),
    }


# --- Sidecar aggregation ---

def aggregate_opening_diagnostics(
    all_game_diagnostics: List[List[dict]],
    diagnostic_end_ply: int,
    extra_plies: int,
    floor_min_ply: int,
    used_floor: bool,
    games_total_iter: int,
) -> dict:
    """Aggregate per-game opening diagnostics into sidecar summary.

    Args:
        all_game_diagnostics: List of per-game diagnostic lists
            (each is a list of per-root records from build_root_diagnostic)
        games_total_iter: Total games in the iteration (may differ from
            len(all_game_diagnostics) if some games had no diagnostics)
    """
    regions = ("near_corner", "edge_band", "interior")

    # Collect records by (ply, color)
    by_ply_color: Dict[Tuple[int, str], List[dict]] = {}
    for game_diags in all_game_diagnostics:
        for rec in game_diags:
            key = (rec["ply"], rec["side_to_move"])
            by_ply_color.setdefault(key, []).append(rec)

    # --- Pass 1: Build per-ply aggregates (no rebound yet) ---
    by_ply: Dict[str, Dict[str, dict]] = {}
    rollup_by_color: Dict[str, List[dict]] = {}

    # Track last penalty-active ply per color
    last_active_ply: Dict[str, int] = {}

    for (ply, color), recs in sorted(by_ply_color.items()):
        n = len(recs)
        ply_key = str(ply)
        if ply_key not in by_ply:
            by_ply[ply_key] = {}

        is_active = recs[0]["penalties_active"]["edge_band"] or recs[0]["penalties_active"]["near_corner"]
        if is_active:
            last_active_ply[color] = ply

        # Mean mass
        mean_raw = {r: sum(rec["raw_mass"][r] for rec in recs) / n for r in regions}
        mean_pen = {r: sum(rec["penalized_mass"][r] for rec in recs) / n for r in regions}
        mean_vis = {r: sum(rec["visit_mass"][r] for rec in recs) / n for r in regions}
        mean_shift = {r: round(mean_pen[r] - mean_raw[r], 4) for r in regions}

        # Top-1 region percentages (all use same denominator n)
        raw_top1_pct = {r: 0 for r in regions}
        pen_top1_pct = {r: 0 for r in regions}
        vis_top1_pct = {r: 0 for r in regions}
        for rec in recs:
            raw_top1_pct[rec["raw_top1"]["primary_region"]] += 1
            pen_top1_pct[rec["penalized_top1"]["primary_region"]] += 1
            vis_top1_pct[rec["visit_top1"]["primary_region"]] += 1
        raw_top1_pct = {r: round(v / n, 3) for r, v in raw_top1_pct.items()}
        pen_top1_pct = {r: round(v / n, 3) for r, v in pen_top1_pct.items()}
        vis_top1_pct = {r: round(v / n, 3) for r, v in vis_top1_pct.items()}

        # Mean legal counts (keep as float with 1 decimal)
        mean_legal = {r: round(sum(rec["legal_move_counts"][r] for rec in recs) / n, 1) for r in regions}

        entry = {
            "n": n,
            "penalties_active": recs[0]["penalties_active"],
            "mean_raw_mass": {r: round(mean_raw[r], 4) for r in regions},
            "mean_penalized_mass": {r: round(mean_pen[r], 4) for r in regions},
            "mean_visit_mass": {r: round(mean_vis[r], 4) for r in regions},
            "mean_penalty_shift": mean_shift,
            "raw_top1_region_pct": raw_top1_pct,
            "penalized_top1_region_pct": pen_top1_pct,
            "visit_top1_region_pct": vis_top1_pct,
            "mean_legal_counts": mean_legal,
        }

        by_ply[ply_key][color] = entry
        rollup_by_color.setdefault(color, []).append(entry)

    # --- Pass 2: Add rebound metrics for post-penalty plies ---
    for (ply, color), recs in sorted(by_ply_color.items()):
        is_active = recs[0]["penalties_active"]["edge_band"] or recs[0]["penalties_active"]["near_corner"]
        if is_active:
            continue  # Rebound only applies to post-penalty plies
        last_ply = last_active_ply.get(color)
        if last_ply is None:
            continue
        last_entry = by_ply.get(str(last_ply), {}).get(color)
        this_entry = by_ply.get(str(ply), {}).get(color)
        if last_entry and this_entry:
            this_entry["rebound_vs_last_active"] = {
                "near_corner_mass_delta": round(
                    this_entry["mean_visit_mass"]["near_corner"] - last_entry["mean_visit_mass"]["near_corner"], 4),
                "edge_band_mass_delta": round(
                    this_entry["mean_visit_mass"]["edge_band"] - last_entry["mean_visit_mass"]["edge_band"], 4),
            }

    # --- Build all_diagnostic_plies rollup (weighted by n) ---
    all_diag = {}
    for color, entries in rollup_by_color.items():
        total_n = sum(e["n"] for e in entries)
        if total_n == 0:
            continue
        agg_entry = {
            "n": total_n,
            "mean_raw_mass": {r: round(sum(e["mean_raw_mass"][r] * e["n"] for e in entries) / total_n, 4) for r in regions},
            "mean_penalized_mass": {r: round(sum(e["mean_penalized_mass"][r] * e["n"] for e in entries) / total_n, 4) for r in regions},
            "mean_visit_mass": {r: round(sum(e["mean_visit_mass"][r] * e["n"] for e in entries) / total_n, 4) for r in regions},
            "mean_penalty_shift": {r: round(sum(e["mean_penalty_shift"][r] * e["n"] for e in entries) / total_n, 4) for r in regions},
            "raw_top1_region_pct": {r: round(sum(e["raw_top1_region_pct"][r] * e["n"] for e in entries) / total_n, 3) for r in regions},
            "penalized_top1_region_pct": {r: round(sum(e["penalized_top1_region_pct"][r] * e["n"] for e in entries) / total_n, 3) for r in regions},
            "visit_top1_region_pct": {r: round(sum(e["visit_top1_region_pct"][r] * e["n"] for e in entries) / total_n, 3) for r in regions},
            "mean_legal_counts": {r: round(sum(e["mean_legal_counts"][r] * e["n"] for e in entries) / total_n, 1) for r in regions},
        }
        all_diag[color] = agg_entry

    return {
        "version": 1,
        "diagnostic_end_ply": diagnostic_end_ply,
        "extra_plies_after_penalty": extra_plies,
        "floor_min_ply": floor_min_ply,
        "used_floor": used_floor,
        "games_total": games_total_iter,
        "games_with_opening_diagnostics": len(all_game_diagnostics),
        "all_diagnostic_plies": all_diag,
        "by_ply": by_ply,
    }
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile scripts/GPU/alphazero/opening_diagnostics.py`
Expected: No output (success)

- [ ] **Step 3: Commit**

```bash
git add scripts/GPU/alphazero/opening_diagnostics.py
git commit -m "feat: add opening_diagnostics module with shared classification and aggregation"
```

---

### Task 2: Integrate diagnostics into the self-play game loop

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py:23` (add import)
- Modify: `scripts/GPU/alphazero/self_play.py:268-321` (add field to GameRecord)
- Modify: `scripts/GPU/alphazero/self_play.py:440-548` (build diagnostic records in game loop)

- [ ] **Step 1: Add import**

At line 23 (existing mcts import), add after it:

```python
from .opening_diagnostics import build_root_diagnostic, compute_diagnostic_end_ply
from .mcts import decode_move
```

- [ ] **Step 2: Add opening_diagnostics field to GameRecord**

After the `rg_top1_samples` field (last field in GameRecord, around line 321), add:

```python
    # Opening penalty diagnostics (per-root records for diagnostic window plies)
    opening_diagnostics: List[dict] = field(default_factory=list)
    opening_diagnostics_meta: Optional[dict] = None
```

Add `Optional` to the typing import if not already present.

- [ ] **Step 3: Build diagnostic records in the game loop**

In `play_game()`, after the MCTS config is resolved (around line 395 where `cfg` is set), compute the diagnostic window:

```python
    # Compute opening diagnostics window
    _diag_end_ply, _diag_used_floor = compute_diagnostic_end_ply(
        cfg.root_edge_band_penalty_ply, cfg.root_near_corner_penalty_ply,
    )
    _opening_diags: List[dict] = []
```

Then inside the game loop, after `visit_counts, root_value, root = mcts.search_from_root(...)` (line 442) and before `move = mcts.select_move(...)` (around line 530), add:

```python
        # Build opening diagnostic record if within window
        if ply < _diag_end_ply and root.priors_raw is not None:
            _opening_diags.append(build_root_diagnostic(
                ply=ply,
                side_to_move=state.to_move,
                visit_counts=visit_counts,
                priors_raw=root.priors_raw,
                priors_adjusted=root.priors,
                board_size=active_size,
                band_width=cfg.root_edge_band_width,
                corner_radius=cfg.root_near_corner_radius,
                edge_penalty=cfg.root_edge_band_penalty,
                corner_penalty=cfg.root_near_corner_penalty,
                edge_penalty_ply=cfg.root_edge_band_penalty_ply,
                corner_penalty_ply=cfg.root_near_corner_penalty_ply,
                decode_fn=decode_move,
            ))
```

Then after the game loop ends, when constructing the GameRecord (around line 570), add the diagnostics:

```python
    game.opening_diagnostics = _opening_diags
    game.opening_diagnostics_meta = {
        "version": 1,
        "diagnostic_end_ply": _diag_end_ply,
        "extra_plies_after_penalty": 2,
        "floor_min_ply": 4,
        "used_floor": _diag_used_floor,
    }
```

- [ ] **Step 4: Verify syntax**

Run: `python3 -m py_compile scripts/GPU/alphazero/self_play.py`

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/self_play.py
git commit -m "feat: build opening diagnostic records in self-play game loop"
```

---

### Task 3: Add diagnostics to IPC messages

**Files:**
- Modify: `scripts/GPU/alphazero/ipc_messages.py:59-97` (add field to GameComplete)

- [ ] **Step 1: Add opening_diagnostics fields to GameComplete**

After the `adj_total_visits` field (last field, around line 96), add:

```python
    # Opening penalty diagnostics (per-root records for diagnostic window)
    opening_diagnostics: Tuple[dict, ...] = ()
    opening_diagnostics_meta: Optional[dict] = None
```

Note: Use `Tuple[dict, ...]` because GameComplete is a frozen dataclass (lists are not hashable).

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile scripts/GPU/alphazero/ipc_messages.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/GPU/alphazero/ipc_messages.py
git commit -m "feat: add opening diagnostics fields to GameComplete IPC message"
```

---

### Task 4: Pass diagnostics from worker to trainer

**Files:**
- Modify: `scripts/GPU/alphazero/self_play_worker.py:196-238` (pack diagnostics into GameComplete)

- [ ] **Step 1: Add diagnostics to GameComplete construction**

In the `stats_queue.put(GameComplete(...))` call (around line 210-238), add after the `adj_total_visits` field:

```python
                opening_diagnostics=tuple(game.opening_diagnostics),
                opening_diagnostics_meta=game.opening_diagnostics_meta,
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile scripts/GPU/alphazero/self_play_worker.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/GPU/alphazero/self_play_worker.py
git commit -m "feat: pass opening diagnostics from worker to trainer via IPC"
```

---

### Task 5: Write diagnostics to per-game JSON files

**Files:**
- Modify: `scripts/GPU/alphazero/game_saver.py:16-114` (accept and write diagnostics)
- Modify: `scripts/GPU/alphazero/game_saver.py:148-186` (pass through in GameSaver class)

- [ ] **Step 1: Add parameters to save_game_replay()**

Add two new parameters after `resigned_by`:

```python
    opening_diagnostics: Optional[list] = None,
    opening_diagnostics_meta: Optional[dict] = None,
```

- [ ] **Step 2: Write diagnostics to the record dict**

After the `record` dict is built (after line 105 `"meta": meta`), add:

```python
    if opening_diagnostics:
        record["opening_diagnostics"] = opening_diagnostics
        record["opening_diagnostics_meta"] = opening_diagnostics_meta
```

- [ ] **Step 3: Add parameters to GameSaver.maybe_save_game()**

Add the same two parameters to `maybe_save_game()`:

```python
    opening_diagnostics: Optional[list] = None,
    opening_diagnostics_meta: Optional[dict] = None,
```

And pass them through to `save_game_replay()`:

```python
        filepath = save_game_replay(
            ...,
            resigned_by=resigned_by,
            opening_diagnostics=opening_diagnostics,
            opening_diagnostics_meta=opening_diagnostics_meta,
        )
```

- [ ] **Step 4: Verify syntax**

Run: `python3 -m py_compile scripts/GPU/alphazero/game_saver.py`

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/game_saver.py
git commit -m "feat: write opening diagnostics to per-game JSON files"
```

---

### Task 6: Pass diagnostics through trainer to game saver

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` — Both game saver call sites + diagnostic collection

- [ ] **Step 1: Add import**

At the top of trainer.py, add:

```python
from .opening_diagnostics import aggregate_opening_diagnostics, compute_diagnostic_end_ply
```

- [ ] **Step 2: Add diagnostics collection variable**

In the parallel code path, where iteration variables are reset (around line 1820-1855), add:

```python
        all_opening_diagnostics = []  # Collect per-game diagnostic lists for sidecar aggregation
```

Do the same in the non-parallel path variable initialization (around line 1114-1152).

- [ ] **Step 3: Pass diagnostics to game_saver in parallel path (Path 1)**

In `process_stats_message` (around line 1266), modify the `game_saver.maybe_save_game()` call to include:

```python
                opening_diagnostics=list(msg.opening_diagnostics) if msg.opening_diagnostics else None,
                opening_diagnostics_meta=msg.opening_diagnostics_meta,
```

Also collect for aggregation, right before the game_saver call:

```python
            # Collect opening diagnostics for sidecar aggregation
            if msg.opening_diagnostics:
                all_opening_diagnostics.append(list(msg.opening_diagnostics))
```

Add `all_opening_diagnostics` to the `nonlocal` declarations in `process_stats_message`.

- [ ] **Step 4: Pass diagnostics to game_saver in non-parallel path (Path 2)**

In the non-parallel game processing (around line 2059-2068), modify the `game_saver.maybe_save_game()` call to include:

```python
                opening_diagnostics=game.opening_diagnostics if game.opening_diagnostics else None,
                opening_diagnostics_meta=game.opening_diagnostics_meta,
```

Also collect:

```python
                # Collect opening diagnostics for sidecar aggregation
                if game.opening_diagnostics:
                    all_opening_diagnostics.append(game.opening_diagnostics)
```

- [ ] **Step 5: Verify syntax**

Run: `python3 -m py_compile scripts/GPU/alphazero/trainer.py`

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py
git commit -m "feat: pass opening diagnostics through trainer to game saver"
```

---

### Task 7: Aggregate diagnostics into the sidecar

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py` — Add aggregation to sidecar write block

- [ ] **Step 1: Add aggregation to the sidecar write block**

In the sidecar write block (after `_sidecar = { ... }` dict is built, around line 2370), add the opening diagnostics section before `games_dir.mkdir(...)`:

```python
            # Aggregate opening diagnostics into sidecar
            if all_opening_diagnostics:
                _diag_end, _diag_floor_used = compute_diagnostic_end_ply(
                    getattr(mcts_config_obj, 'root_edge_band_penalty_ply', 0),
                    getattr(mcts_config_obj, 'root_near_corner_penalty_ply', 0),
                )
                _sidecar["opening_penalty_diagnostics"] = aggregate_opening_diagnostics(
                    all_game_diagnostics=all_opening_diagnostics,
                    diagnostic_end_ply=_diag_end,
                    extra_plies=2,
                    floor_min_ply=4,
                    used_floor=_diag_floor_used,
                    games_total_iter=games_generated,
                )
```

Note: Find the correct variable name for the MCTSConfig object in scope. It may be `mcts_config` or constructed from CLI args. Check the exact variable name at the sidecar write point.

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile scripts/GPU/alphazero/trainer.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/GPU/alphazero/trainer.py
git commit -m "feat: aggregate opening diagnostics into per-iteration stats sidecar"
```

---

### Task 8: Verify end-to-end

- [ ] **Step 1: Syntax check all modified files**

```bash
python3 -m py_compile scripts/GPU/alphazero/opening_diagnostics.py
python3 -m py_compile scripts/GPU/alphazero/self_play.py
python3 -m py_compile scripts/GPU/alphazero/ipc_messages.py
python3 -m py_compile scripts/GPU/alphazero/self_play_worker.py
python3 -m py_compile scripts/GPU/alphazero/game_saver.py
python3 -m py_compile scripts/GPU/alphazero/trainer.py
```

- [ ] **Step 2: Run a short training session (2-3 iterations with opening penalties enabled)**

```bash
python -m scripts.GPU.alphazero.train \
  --iterations 3 \
  --games-per-iter 10 \
  --root-edge-band-penalty 1.5 \
  --root-edge-band-penalty-ply 16 \
  --root-edge-band-width 2 \
  --root-near-corner-penalty 2.0 \
  --root-near-corner-penalty-ply 16 \
  --root-near-corner-radius 3 \
  --save-games
```

- [ ] **Step 3: Verify per-game JSON has opening_diagnostics**

```bash
python3 -c "
import json
d = json.load(open('scripts/GPU/logs/games/iter_0001_game_000.json'))
diags = d.get('opening_diagnostics', [])
meta = d.get('opening_diagnostics_meta', {})
print(f'Meta: {json.dumps(meta, indent=2)}')
print(f'Records: {len(diags)}')
if diags:
    print(f'First record: {json.dumps(diags[0], indent=2)}')
"
```

Expected: `opening_diagnostics` array with records for plies 0 through diagnostic_end_ply-1. Each record has `raw_mass`, `penalized_mass`, `visit_mass`, `*_top1` entries.

- [ ] **Step 4: Verify sidecar has opening_penalty_diagnostics**

```bash
python3 -c "
import json
d = json.load(open('scripts/GPU/logs/games/iter_0001_stats.json'))
opd = d.get('opening_penalty_diagnostics', {})
print(f'Version: {opd.get(\"version\")}')
print(f'diagnostic_end_ply: {opd.get(\"diagnostic_end_ply\")}')
print(f'games_total: {opd.get(\"games_total\")}')
print(f'Plies in by_ply: {sorted(opd.get(\"by_ply\", {}).keys())}')
if opd.get('all_diagnostic_plies'):
    for color, data in opd['all_diagnostic_plies'].items():
        print(f'{color}: n={data[\"n\"]}, mean_raw_mass={data[\"mean_raw_mass\"]}')
"
```

Expected: Aggregated summaries with non-zero mass values, correct ply range, and per-color breakdowns.

- [ ] **Step 5: Verify mass sums to ~1.0**

```bash
python3 -c "
import json
d = json.load(open('scripts/GPU/logs/games/iter_0001_game_000.json'))
for rec in d.get('opening_diagnostics', [])[:3]:
    for stage in ('raw_mass', 'penalized_mass', 'visit_mass'):
        total = sum(rec[stage].values())
        ok = '✓' if 0.99 < total < 1.01 else '✗'
        print(f'ply={rec[\"ply\"]} {stage} sum={total:.4f} {ok}')
"
```

Expected: All sums between 0.99 and 1.01.

- [ ] **Step 6: Verify rebound metrics appear for post-penalty plies**

```bash
python3 -c "
import json
d = json.load(open('scripts/GPU/logs/games/iter_0001_stats.json'))
opd = d.get('opening_penalty_diagnostics', {})
for ply_key, colors in opd.get('by_ply', {}).items():
    for color, data in colors.items():
        if 'rebound_vs_last_active' in data:
            print(f'ply={ply_key} {color}: rebound={data[\"rebound_vs_last_active\"]}')
"
```

Expected: Rebound entries for plies after the penalty window, with non-zero deltas.

---

## Implementation Notes

- **Single classification helper**: `opening_diagnostics.py` is the single source of truth. Both `build_root_diagnostic()` (per-game) and `aggregate_opening_diagnostics()` (sidecar) use the same `classify_move()` / `primary_region()` functions. Zero drift by construction.

- **Key normalization**: `build_root_diagnostic()` normalizes all prior dicts to `(row, col)` keys at the top via `_normalize_priors_to_rc()`. This eliminates the fragile seam between `visit_counts` (tuple keys) and priors (int move_id keys).

- **Post-root-adjustment priors**: In all code comments, refer to `root.priors` as "post-root-adjustment priors" (includes Dirichlet noise + penalties). JSON fields use `penalized_*` for readability.

- **Top-1 tie handling**: Ties resolve by first-encountered move (dict iteration order). This is acceptable for v1. Documented in `build_root_diagnostic()` docstring.

- **Frozen dataclass constraint**: `GameComplete` in `ipc_messages.py` is a frozen dataclass. Use `Tuple[dict, ...]` for the diagnostics list (not `List`), since lists aren't hashable.

- **priors_raw safety**: The game loop checks `root.priors_raw is not None` before building a diagnostic record, avoiding partial garbage if any code path leaves it unset.

- **MCTSConfig variable name**: At the sidecar write point in trainer.py, verify the exact variable name for the MCTSConfig object. It may be constructed from CLI args and passed to the self-play functions. Check what's in scope at the sidecar write block.

- **Two-pass rebound**: Aggregation builds all per-ply entries first, then adds `rebound_vs_last_active` in a second pass. This avoids depending on insertion order.

- **Backward compatibility**: Old game JSON files without `opening_diagnostics` will simply lack the key. The replay analyzer already handles missing keys gracefully.
