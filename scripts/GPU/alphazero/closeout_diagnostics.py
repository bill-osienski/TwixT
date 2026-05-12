"""Inline closeout root diagnostics (spec 2026-05-03 §8).

Composes closeout-specific sub-blocks (goal_completion,
endpoint_completion_ranking, distance_reducing_ranking,
selected_move_classification) into per-ply diagnostic records used by
self_play.py during the closeout window. Does NOT extend or call
opening_diagnostics.build_root_diagnostic — the per-record schema in
spec §8.3 doesn't need the region/policy mass distributions that
build_root_diagnostic produces.

Design split:
  - build_closeout_diagnostic_partial: pre-move-selection portion (no
    selected_move). Computes root_summary, goal_completion sub-block,
    and per-completion-move policy/visit ranking.
  - finalize_closeout_diagnostic: post-move-selection portion. Adds
    selected_move + selected_move_classification via the connectivity
    helper.

Capture is BEST-EFFORT; callers wrap with try/except. See spec §4.3.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple


def _rank_moves_by_score(
    visit_counts: Dict[Tuple[int, int], int],
    priors_raw,
    decode_fn,
    board_size: int,
):
    """Return (move_to_visit_rank, move_to_policy_rank, move_to_policy_prob, total_visits).

    Ranks are 1-based; ties broken by lexicographic move order for determinism.

    `priors_raw` is the MCTS root's per-legal-move prior dict ({move_id: prob}),
    matching the format mcts._expand_batch writes. Earlier code assumed a
    length-n_cells flat array and silently produced empty ranks against the
    real dict format — leaving best_policy_rank=None on every closeout record.
    """
    move_to_visits = dict(visit_counts)
    sorted_by_visits = sorted(
        move_to_visits.items(), key=lambda kv: (-kv[1], kv[0])
    )
    move_to_visit_rank = {m: i + 1 for i, (m, _) in enumerate(sorted_by_visits)}
    total_visits = sum(move_to_visits.values())

    move_to_policy_rank: Dict[Tuple[int, int], int] = {}
    move_to_policy_prob: Dict[Tuple[int, int], float] = {}
    if priors_raw:
        scored = []
        for mid, p in priors_raw.items():
            p = float(p)
            if p <= 0.0:
                continue
            mv = decode_fn(mid)
            scored.append((mv, p))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        for i, (m, p) in enumerate(scored):
            move_to_policy_rank[m] = i + 1
            move_to_policy_prob[m] = p
    return move_to_visit_rank, move_to_policy_rank, move_to_policy_prob, total_visits


def _ranking_block(
    candidate_moves: List[Tuple[int, int]],
    move_to_visit_rank: Dict[Tuple[int, int], int],
    move_to_policy_rank: Dict[Tuple[int, int], int],
    move_to_policy_prob: Dict[Tuple[int, int], float],
    visit_counts: Dict[Tuple[int, int], int],
    total_visits: int,
) -> Optional[dict]:
    """Compute a ranking sub-block over the given candidate moves."""
    if not candidate_moves:
        return None
    visit_ranks = [move_to_visit_rank.get(m) for m in candidate_moves]
    visit_ranks = [r for r in visit_ranks if r is not None]
    policy_ranks = [move_to_policy_rank.get(m) for m in candidate_moves]
    policy_ranks = [r for r in policy_ranks if r is not None]
    if not visit_ranks and not policy_ranks:
        return None
    best_visit_rank = min(visit_ranks) if visit_ranks else None
    best_visit_count = max(
        (visit_counts.get(m, 0) for m in candidate_moves), default=0
    )
    best_visit_share = (
        best_visit_count / total_visits if total_visits > 0 else None
    )
    best_policy_rank = min(policy_ranks) if policy_ranks else None
    best_policy_prob = max(
        (move_to_policy_prob.get(m, 0.0) for m in candidate_moves), default=0.0
    )
    return {
        "best_policy_rank":            best_policy_rank,
        "best_policy_prob":            float(best_policy_prob),
        "best_visit_rank":             best_visit_rank,
        "best_visit_share":            best_visit_share,
        "best_completion_visit_share": best_visit_share,
        "any_in_policy_top5":          any(r is not None and r <= 5 for r in policy_ranks),
        "any_in_visit_top5":           any(r is not None and r <= 5 for r in visit_ranks),
    }


def build_closeout_diagnostic_partial(
    ply: int,
    side_to_move: str,
    visit_counts: Dict[Tuple[int, int], int],
    priors_raw,
    priors_adjusted,
    root,
    goal_completion_state: dict,
    board_size: int,
    skip_distance_reducing: bool,
    decode_fn,
) -> dict:
    """Build a partial closeout diagnostic record (pre-move-selection).

    Returns a dict with root_summary, goal_completion sub-block, and
    endpoint/distance-reducing rankings. selected_move and classification
    are added later by finalize_closeout_diagnostic.

    `priors_adjusted` is reserved for future use (the root's noise-added
    priors) but is not currently consumed — kept in the signature so
    callers don't break when adjusted-priors features land.
    """
    move_to_visit_rank, move_to_policy_rank, move_to_policy_prob, total_visits = (
        _rank_moves_by_score(visit_counts, priors_raw, decode_fn, board_size)
    )

    completion_moves = [tuple(m) for m in (goal_completion_state.get("endpoint_completion_moves") or [])]
    if skip_distance_reducing:
        reducing_moves: List[Tuple[int, int]] = []
    else:
        reducing_moves = [tuple(m) for m in (goal_completion_state.get("distance_reducing_moves") or [])]

    endpoint_block = _ranking_block(
        completion_moves, move_to_visit_rank, move_to_policy_rank,
        move_to_policy_prob, visit_counts, total_visits,
    )
    reducing_block = _ranking_block(
        reducing_moves, move_to_visit_rank, move_to_policy_rank,
        move_to_policy_prob, visit_counts, total_visits,
    )

    return {
        "ply": ply,
        "side_to_move": side_to_move,
        "active_size": board_size,
        "root_summary": {
            "visit_count": int(getattr(root, "visit_count", 0)),
            "q_value":     float(getattr(root, "q_value", 0.0) or 0.0),
            "nn_value":    float(getattr(root, "nn_value", 0.0) or 0.0),
        },
        "goal_completion": {
            "max_depth":                  goal_completion_state.get("max_depth"),
            "total_goal_distance_before": goal_completion_state.get("total_goal_distance"),
            "endpoint_distances":         dict(goal_completion_state.get("endpoint_distances") or {}),
            "largest_component_size":     goal_completion_state.get("largest_component_size"),
            "category":                   goal_completion_state.get("category"),
            "endpoint_completion_moves":  [list(m) for m in completion_moves],
            "distance_reducing_moves":    None if skip_distance_reducing else [list(m) for m in reducing_moves],
        },
        "endpoint_completion_ranking": endpoint_block,
        "distance_reducing_ranking":   reducing_block,
        # selected_move + selected_move_classification added by finalize.
    }


def finalize_closeout_diagnostic(
    partial_diag: dict,
    state_before,
    player: str,
    selected_move: Tuple[int, int],
    goal_state_before: dict,
) -> dict:
    """Add selected_move + selected_move_classification to a partial record.

    Calls connectivity_diagnostics.classify_selected_conversion_move using
    the pre-move state and the captured goal-completion state. Returns a
    new dict (does not mutate `partial_diag`).
    """
    from .connectivity_diagnostics import classify_selected_conversion_move

    classification = classify_selected_conversion_move(
        state_before, player, selected_move, goal_state_before
    )
    out = dict(partial_diag)
    out["selected_move"] = list(selected_move)
    out["selected_move_classification"] = classification
    return out
