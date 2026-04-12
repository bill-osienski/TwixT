"""Opening penalty diagnostics — shared helpers.

This module is the single source of truth for:
- Regional classification of moves (edge_band, near_corner, interior)
- Per-root diagnostic record building
- Sidecar aggregation of per-game records

Both per-game records and sidecar aggregates use the same classification
functions, ensuring zero drift between raw records and summaries.
"""
from __future__ import annotations

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
        used_floor is True if max(edge_penalty_ply, corner_penalty_ply) < floor.
    """
    max_penalty_ply = max(edge_penalty_ply, corner_penalty_ply)
    used_floor = max_penalty_ply < floor
    return max(max_penalty_ply, floor) + extra, used_floor


# --- Key normalization ---

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


# --- Per-root record builder ---

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

    Two-pass approach:
    - Pass 1: Build all per-ply entries
    - Pass 2: Add rebound_vs_last_active for post-penalty plies

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
    # Rebound compares post-search behavior (visit_mass) after penalties end
    # to the last active post-search behavior.
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
