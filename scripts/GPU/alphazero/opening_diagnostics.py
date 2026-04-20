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


# --- Canonical early-override penalty selector ---
#
# This is the single source of truth for "what near-corner penalty applies at
# this ply?". Both the MCTS runtime (`_add_dirichlet_noise`) and the diagnostic
# record builder read through this function so the stored
# `effective_near_corner_penalty` value is guaranteed to match what MCTS
# actually applied.

def effective_near_corner_penalty(
    ply: int,
    corner_penalty: float,
    corner_penalty_ply: int,
    corner_penalty_early: float,
    corner_penalty_early_plies: int,
) -> float:
    """Return the near-corner penalty λ to apply at this ply (0.0 = none).

    Precedence:
      1. Early override: if `corner_penalty_early > 0` and
         `corner_penalty_early_plies > 0` and `ply < corner_penalty_early_plies`,
         use the early value.
      2. Baseline: else if `corner_penalty > 0` and `corner_penalty_ply > 0`
         and `ply < corner_penalty_ply`, use the baseline value.
      3. None: 0.0 (no penalty applied).

    The early window is NOT required to be a subset of the baseline window —
    e.g. an early-only run with baseline disabled uses only the early branch.
    """
    if (
        corner_penalty_early > 0.0
        and corner_penalty_early_plies > 0
        and ply < corner_penalty_early_plies
    ):
        return corner_penalty_early
    if (
        corner_penalty > 0.0
        and corner_penalty_ply > 0
        and ply < corner_penalty_ply
    ):
        return corner_penalty
    return 0.0


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
    corner_penalty_early: float = 0.0,
    corner_penalty_early_plies: int = 0,
) -> dict:
    """Build a single per-root diagnostic record.

    All inputs are normalized to (row, col) keys internally.

    Args:
        visit_counts: Dict[(row, col) -> visits] from MCTS
        priors_raw: Dict[move_id or (r,c) -> prior] — raw NN output (root.priors_raw)
        priors_adjusted: Dict[move_id or (r,c) -> prior] — post-root-adjustment
            (includes Dirichlet noise + penalties, not penalty-only)
        decode_fn: Function move_id -> (r, c) for key normalization
        corner_penalty_early: Phase 2 early-override penalty λ_early. When >0
            AND `corner_penalty_early_plies > 0` AND `ply < early_plies`, this
            value is the one MCTS actually applied at the root.
        corner_penalty_early_plies: Phase 2 early-override window end.

    Note: top-1 ties resolve by first-encountered move (iteration order).
    """
    # Normalize all priors to (row, col) keys for uniform processing
    raw_rc = _normalize_priors_to_rc(priors_raw, decode_fn)
    adj_rc = _normalize_priors_to_rc(priors_adjusted, decode_fn)

    edge_active = edge_penalty > 0 and edge_penalty_ply > 0 and ply < edge_penalty_ply
    # Phase 2: resolve the effective near-corner penalty actually applied at
    # this ply. `corner_active` reflects whether the penalty has a non-zero
    # effect, not which window was used — that distinction lives in the
    # returned `near_corner_penalty_source` field below.
    effective_corner_pen = effective_near_corner_penalty(
        ply=ply,
        corner_penalty=corner_penalty,
        corner_penalty_ply=corner_penalty_ply,
        corner_penalty_early=corner_penalty_early,
        corner_penalty_early_plies=corner_penalty_early_plies,
    )
    corner_active = effective_corner_pen > 0 and corner_radius > 0
    if (
        corner_penalty_early > 0.0
        and corner_penalty_early_plies > 0
        and ply < corner_penalty_early_plies
    ):
        corner_penalty_source = "early"
    elif corner_active:
        corner_penalty_source = "baseline"
    else:
        corner_penalty_source = "none"

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
        # Required #2: the exact penalty MCTS applied at this ply — comes
        # straight from the canonical selector, so no ambiguity about which
        # window was active. 0.0 means no near-corner penalty at this ply.
        "effective_near_corner_penalty": round(effective_corner_pen, 6),
        "near_corner_penalty_source": corner_penalty_source,  # "early" | "baseline" | "none"
        "config": {
            "edge_band_width": band_width,
            "edge_band_penalty": edge_penalty,
            "edge_band_penalty_ply": edge_penalty_ply,
            "near_corner_radius": corner_radius,
            "near_corner_penalty": corner_penalty,
            "near_corner_penalty_ply": corner_penalty_ply,
            # Required #1: echo the early-override config so each record is
            # self-describing. Zero values indicate the override was not in
            # use for this run.
            "near_corner_penalty_early": corner_penalty_early,
            "near_corner_penalty_early_plies": corner_penalty_early_plies,
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


# --- Root-child diagnostic builder (Phase 1: ply 0-1 deep inspection) ---

def build_root_child_details(
    root,
    c_puct: float,
    board_size: int,
    band_width: int,
    corner_radius: int,
    top_k: int,
    decode_fn,
) -> dict:
    """Build deep per-child diagnostics at the root for early-ply analysis.

    Captures exactly why the search chose what it chose:
    - Per-child: prior_raw, prior_penalized, visits, q, u, score, region
    - Root summary: best_by_{penalized_prior, visit, score, q, u}
    - Score-tie detection (how many children are within eps of the max score)

    This is diagnostic-only. It is not part of the search; it reads root
    state AFTER all simulations have completed.

    Args:
        root: An MCTSNode-like object with `priors`, `priors_raw`, `children`,
              `visit_count`, `q_value`, `nn_value` attributes. Children are
              MCTSNode-like with `visit_count` and `q_value`.
        c_puct: PUCT exploration constant (from MCTSConfig.c_puct).
        top_k: Number of children to include in `top_children` (sorted by
               visits desc, then score desc).
        decode_fn: Function mapping move_id -> (row, col).

    Returns:
        dict with keys:
            root_summary: {visit_count, q_value, nn_value, c_puct,
                           best_by_{penalized_prior, visit, score, q, u},
                           score_tie_count, n_children_tracked}
            top_children: list of per-child dicts (len <= top_k)
    """
    eps = 1e-8

    priors_raw = root.priors_raw or {}
    priors_adj = root.priors or {}
    children = getattr(root, "children", {}) or {}

    sqrt_parent = math.sqrt(root.visit_count + 1)

    # Build entries for every move in penalized priors (iteration basis for PUCT).
    entries: List[dict] = []
    for mid, p_adj in priors_adj.items():
        r, c = decode_fn(mid)
        p_raw = float(priors_raw.get(mid, 0.0))
        p_adj = float(p_adj)

        child = children.get(mid)
        if child is not None and child.visit_count > 0:
            child_q = float(child.q_value)
            child_visits = int(child.visit_count)
            q = -child_q  # From parent perspective (same as PUCT uses)
        else:
            child_q = 0.0
            child_visits = 0
            q = 0.0

        u = c_puct * p_adj * sqrt_parent / (1 + child_visits)
        score = q + u

        cls = classify_move(r, c, board_size, band_width, corner_radius)

        entries.append({
            "mid": mid,
            "move": [int(r), int(c)],
            "region": cls["primary_region"],
            "is_edge_band": cls["is_edge_band"],
            "is_near_corner": cls["is_near_corner"],
            "prior_raw": round(p_raw, 5),
            "prior_penalized": round(p_adj, 5),
            "visit_count": child_visits,
            "q_value_child": round(child_q, 4),
            "q": round(q, 4),
            "u": round(u, 5),
            "score": round(score, 4),
        })

    total_visits = sum(e["visit_count"] for e in entries)
    total_visits_f = float(max(total_visits, 1))
    for e in entries:
        e["visit_share"] = round(e["visit_count"] / total_visits_f, 4)

    def _best(key):
        if not entries:
            return None
        return max(entries, key=lambda e: e[key])

    def _best_entry(e):
        if e is None:
            return None
        return {
            "move": e["move"],
            "region": e["region"],
            "prior_raw": e["prior_raw"],
            "prior_penalized": e["prior_penalized"],
            "visit_count": e["visit_count"],
            "visit_share": e["visit_share"],
            "q": e["q"],
            "u": e["u"],
            "score": e["score"],
        }

    best_prior = _best("prior_penalized")
    best_visit = _best("visit_count")
    best_score = _best("score")
    best_q = _best("q")
    best_u = _best("u")

    # Score tie: count entries whose score is within eps of the max
    score_tie_count = 0
    if best_score is not None:
        best_s = best_score["score"]
        score_tie_count = sum(1 for e in entries if abs(e["score"] - best_s) <= eps)

    # Sort by visits desc (tiebreak score desc) for top_k
    entries_sorted = sorted(entries, key=lambda e: (-e["visit_count"], -e["score"]))
    top = entries_sorted[:max(0, int(top_k))]

    # Mark score-ties on the selected top slice and drop internal mid key
    best_s = best_score["score"] if best_score is not None else None
    top_out: List[dict] = []
    for e in top:
        rec = {k: v for k, v in e.items() if k != "mid"}
        rec["in_score_tie"] = (
            best_s is not None and abs(e["score"] - best_s) <= eps
        )
        top_out.append(rec)

    # nn_value may be None on unexpanded roots (shouldn't happen at ply>=0,
    # but be defensive)
    nn_value = root.nn_value
    nn_value_out = round(float(nn_value), 4) if nn_value is not None else None

    root_summary = {
        "visit_count": int(root.visit_count),
        "q_value": round(float(root.q_value), 4) if root.visit_count > 0 else 0.0,
        "nn_value": nn_value_out,
        "c_puct": float(c_puct),
        "n_children_tracked": len(entries),
        "score_tie_count": int(score_tie_count),
        "best_by_penalized_prior": _best_entry(best_prior),
        "best_by_visit": _best_entry(best_visit),
        "best_by_score": _best_entry(best_score),
        "best_by_q": _best_entry(best_q),
        "best_by_u": _best_entry(best_u),
    }

    return {
        "root_summary": root_summary,
        "top_children": top_out,
    }


def aggregate_root_child_details(
    all_game_diagnostics: List[List[dict]],
    child_detail_max_ply: int,
) -> dict:
    """Aggregate root_summary.best_by_* across games into a compact rollup.

    Only looks at records that include a `root_summary` key (i.e. records built
    with child details enabled). Produces per-(ply, color) region counts for
    each best-move metric, plus score-tie statistics.

    Args:
        all_game_diagnostics: list of per-game diagnostic record lists.
        child_detail_max_ply: plies [0, child_detail_max_ply) are expected to
            carry child details. Used to filter records.

    Returns:
        {
          "by_ply": {
            "<ply>": {
              "red"|"black": {
                  "n": N,
                  "best_by_<metric>_region_pct": {"near_corner": x, "edge_band": y, "interior": z},
                  ... for each metric ...
                  "score_tie_count_mean": float,
                  "score_tie_rate": float,  # fraction with tie_count > 1
                  "nn_value_mean": float|None,
              }
            }
          },
          "metrics": [list of metric names],
        }
    """
    regions = ("near_corner", "edge_band", "interior")
    metrics = ("best_by_penalized_prior", "best_by_visit",
               "best_by_score", "best_by_q", "best_by_u")

    # Collect records with child details by (ply, color)
    by_key: Dict[Tuple[int, str], List[dict]] = {}
    for game_diags in all_game_diagnostics:
        for rec in game_diags:
            if "root_summary" not in rec:
                continue
            ply = rec["ply"]
            if ply >= child_detail_max_ply:
                continue
            key = (ply, rec["side_to_move"])
            by_key.setdefault(key, []).append(rec)

    by_ply: Dict[str, Dict[str, dict]] = {}
    for (ply, color), recs in sorted(by_key.items()):
        n = len(recs)
        if n == 0:
            continue
        entry: Dict = {"n": n}

        # Region counts per best_by_* metric
        for metric in metrics:
            counts = {r: 0 for r in regions}
            for rec in recs:
                bm = rec["root_summary"].get(metric)
                if bm is None:
                    continue
                r = bm.get("region", "interior")
                if r in counts:
                    counts[r] += 1
            entry[f"{metric}_region_pct"] = {
                r: round(counts[r] / n, 3) for r in regions
            }

        # Score tie statistics
        tie_counts = [int(rec["root_summary"].get("score_tie_count", 1)) for rec in recs]
        entry["score_tie_count_mean"] = round(sum(tie_counts) / n, 3)
        entry["score_tie_rate"] = round(
            sum(1 for t in tie_counts if t > 1) / n, 3
        )

        # nn_value mean (robust to None)
        nn_vals = [
            rec["root_summary"].get("nn_value")
            for rec in recs
            if rec["root_summary"].get("nn_value") is not None
        ]
        entry["nn_value_mean"] = (
            round(sum(nn_vals) / len(nn_vals), 4) if nn_vals else None
        )

        by_ply.setdefault(str(ply), {})[color] = entry

    return {
        "metrics": list(metrics),
        "by_ply": by_ply,
    }


# --- Phase 2: early-override summary ---

def build_early_override_summary(
    opd_aggregate: dict,
    rcd_aggregate: Optional[dict] = None,
    early_plies: int = 2,
) -> dict:
    """Compact ply 0..early_plies-1 summary focused on override effectiveness.

    Pulls mass numbers from the opening-penalty aggregate and best-by-* signals
    from the root-child aggregate, so one compact block answers: did the early
    override actually push visit_mass and the final visit choice away from
    near-corner, and which search component (q / u / score) is still carrying
    corners if anything is?

    Args:
        opd_aggregate: output of `aggregate_opening_diagnostics` (or a sidecar
            `opening_penalty_diagnostics` block — same shape).
        rcd_aggregate: output of `aggregate_root_child_details` (or a sidecar
            `root_child_diagnostics` block). Optional — if absent, the
            disagreement deltas from root_child are omitted.
        early_plies: number of leading plies to include. Defaults to 2, matching
            the CHILD_DETAIL_PLIES constant the trainer uses.

    Returns:
        {
          "early_plies": int,
          "config": run-level near-corner config echoed from opd.run_config,
          "by_ply": {
            "0": {"red": {...}, "black": {...}},
            "1": {...},
          }
        }
        Each per-(ply, color) entry carries:
          n, effective_near_corner_penalty, near_corner_penalty_source,
          raw_near_corner_mass, penalized_near_corner_mass, visit_near_corner_mass,
          visit_minus_penalized_near_corner_mass,
          [best_by_{penalized_prior,visit,score,q,u}_near_corner_pct,
           q_minus_penalized_near_corner, u_minus_penalized_near_corner,
           score_minus_penalized_near_corner, visit_minus_penalized_near_corner_pct,
           score_tie_rate, nn_value_mean]   ← root-child pieces only if rcd is given
    """
    out: dict = {
        "early_plies": int(early_plies),
        "config": dict(opd_aggregate.get("run_config") or {}),
        "by_ply": {},
    }

    opd_by_ply = (opd_aggregate.get("by_ply") or {})
    rcd_by_ply = ((rcd_aggregate or {}).get("by_ply") or {}) if rcd_aggregate else {}

    for ply in range(early_plies):
        ply_str = str(ply)
        opd_entry_by_color = opd_by_ply.get(ply_str) or {}
        rcd_entry_by_color = rcd_by_ply.get(ply_str) or {}
        if not opd_entry_by_color:
            continue
        out["by_ply"][ply_str] = {}

        for color in ("red", "black"):
            opd_entry = opd_entry_by_color.get(color)
            if not opd_entry:
                continue
            # Mass-based signal (distribution of probability mass on near-corner)
            raw_nc = (opd_entry.get("mean_raw_mass") or {}).get("near_corner", 0.0)
            pen_nc = (opd_entry.get("mean_penalized_mass") or {}).get("near_corner", 0.0)
            vis_nc = (opd_entry.get("mean_visit_mass") or {}).get("near_corner", 0.0)

            # Effective penalty at this ply — taken from the first
            # (ply, color) aggregate entry's echo. Since every root at a given
            # ply saw the same penalty, a single value is authoritative.
            eff_pen = opd_entry.get("effective_near_corner_penalty")
            pen_src = opd_entry.get("near_corner_penalty_source")

            entry: dict = {
                "n": int(opd_entry.get("n", 0)),
                "effective_near_corner_penalty": eff_pen,
                "near_corner_penalty_source": pen_src,
                "raw_near_corner_mass": round(float(raw_nc), 4),
                "penalized_near_corner_mass": round(float(pen_nc), 4),
                "visit_near_corner_mass": round(float(vis_nc), 4),
                "visit_minus_penalized_near_corner_mass": round(float(vis_nc) - float(pen_nc), 4),
            }

            # Best-by-* signal: augment only when root-child aggregate is
            # available (the two aggregates cover the same plies when both
            # were emitted by the same run).
            rcd_entry = rcd_entry_by_color.get(color) if rcd_entry_by_color else None
            if rcd_entry:
                pen_pct = (rcd_entry.get("best_by_penalized_prior_region_pct") or {}).get("near_corner", 0.0)
                vis_pct = (rcd_entry.get("best_by_visit_region_pct") or {}).get("near_corner", 0.0)
                score_pct = (rcd_entry.get("best_by_score_region_pct") or {}).get("near_corner", 0.0)
                q_pct = (rcd_entry.get("best_by_q_region_pct") or {}).get("near_corner", 0.0)
                u_pct = (rcd_entry.get("best_by_u_region_pct") or {}).get("near_corner", 0.0)
                entry.update({
                    "best_by_penalized_prior_near_corner_pct": round(float(pen_pct), 3),
                    "best_by_visit_near_corner_pct": round(float(vis_pct), 3),
                    "best_by_score_near_corner_pct": round(float(score_pct), 3),
                    "best_by_q_near_corner_pct": round(float(q_pct), 3),
                    "best_by_u_near_corner_pct": round(float(u_pct), 3),
                    "visit_minus_penalized_near_corner_pct": round(float(vis_pct) - float(pen_pct), 3),
                    "score_minus_penalized_near_corner": round(float(score_pct) - float(pen_pct), 3),
                    "q_minus_penalized_near_corner": round(float(q_pct) - float(pen_pct), 3),
                    "u_minus_penalized_near_corner": round(float(u_pct) - float(pen_pct), 3),
                    "score_tie_rate": rcd_entry.get("score_tie_rate"),
                    "nn_value_mean": rcd_entry.get("nn_value_mean"),
                })

            out["by_ply"][ply_str][color] = entry

    return out


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

        # Forward the per-ply effective penalty. Scan ALL records (not just
        # recs[0]) for the first non-None value: when the analyzer is run
        # over a directory that mixes pre-Phase-2 games (missing the field)
        # with Phase-2 games (carrying it), recs[0] can land on an old record
        # and the bucket would spuriously report None. Any Phase-2 record in
        # the bucket implies the field is "known" for that ply.
        _eff_pen = next(
            (r.get("effective_near_corner_penalty") for r in recs
             if r.get("effective_near_corner_penalty") is not None),
            None,
        )
        _eff_src = next(
            (r.get("near_corner_penalty_source") for r in recs
             if r.get("near_corner_penalty_source") is not None),
            None,
        )

        entry = {
            "n": n,
            "penalties_active": recs[0]["penalties_active"],
            "effective_near_corner_penalty": _eff_pen,
            "near_corner_penalty_source": _eff_src,
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

    # Pull the run-level config echo from a representative per-root record.
    # For a homogeneous analyzer run, every record's `config` block is
    # identical. For mixed inputs (e.g. analyzer scanning all sidecars from
    # `scripts/GPU/logs/games`), some games predate Phase 2 and their config
    # lacks the `near_corner_penalty_early*` fields. Prefer a config that
    # carries the Phase 2 keys so the echo is actionable; fall back to any
    # non-empty config otherwise.
    _run_config: Dict[str, float] = {}
    _fallback_config: Dict[str, float] = {}
    for gd in all_game_diagnostics:
        if not gd:
            continue
        _cfg = (gd[0] or {}).get("config") or {}
        if not _cfg:
            continue
        if not _fallback_config:
            _fallback_config = dict(_cfg)
        if "near_corner_penalty_early" in _cfg:
            _run_config = dict(_cfg)
            break
    if not _run_config:
        _run_config = _fallback_config

    return {
        # Bumped to v2: records now carry `effective_near_corner_penalty`,
        # `near_corner_penalty_source`, and an expanded `config` with the
        # Phase 2 early-override fields. The aggregate itself adds `run_config`.
        "version": 2,
        "diagnostic_end_ply": diagnostic_end_ply,
        "extra_plies_after_penalty": extra_plies,
        "floor_min_ply": floor_min_ply,
        "used_floor": used_floor,
        "games_total": games_total_iter,
        "games_with_opening_diagnostics": len(all_game_diagnostics),
        "run_config": _run_config,
        "all_diagnostic_plies": all_diag,
        "by_ply": by_ply,
    }
