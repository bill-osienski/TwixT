"""Connectivity-aware replay diagnostics — Phase 1 of the retrain design spec.

Computes per-position Twixt-structural stats (goal-touching components,
largest component size, etc.) from game JSON move histories, then aggregates
by ply bucket + outcome for analyzer-side reporting.
"""
from __future__ import annotations
from typing import Dict, List
from collections import defaultdict

from .game.twixt_state import TwixtState


def compute_position_connectivity(state: TwixtState) -> Dict[str, object]:
    """Per-position connectivity stats using the shared connectivity_masks helper."""
    out: Dict[str, object] = {}

    for player, prefix, goal1_name, goal2_name in (
        ("red", "red", "top", "bottom"),
        ("black", "black", "left", "right"),
    ):
        m_g1, m_g2, m_both = state.connectivity_masks(player)
        out[f"{prefix}_has_{goal1_name}_component"] = bool(m_g1.sum() > 0)
        out[f"{prefix}_has_{goal2_name}_component"] = bool(m_g2.sum() > 0)

        # Largest component size
        pegs_of = [(r, c) for (r, c), col in state.pegs.items() if col == player]
        seen = set()
        sizes = []
        for peg in pegs_of:
            if peg in seen:
                continue
            comp = state._get_connected_component(peg, player)
            sizes.append(len(comp))
            seen.update(comp)
        out[f"{prefix}_largest_component_size"] = max(sizes) if sizes else 0

        # Number of goal-touching components (0, 1, or 2)
        # Recount components that touch any goal edge
        seen = set()
        touching_count = 0
        for peg in pegs_of:
            if peg in seen:
                continue
            comp = state._get_connected_component(peg, player)
            seen.update(comp)
            if player == "red":
                touches = any(r == 0 or r == state.active_size - 1 for (r, _) in comp)
            else:
                touches = any(c == 0 or c == state.active_size - 1 for (_, c) in comp)
            if touches:
                touching_count += 1
        out[f"{prefix}_n_goal_touching_components"] = min(touching_count, 2)

    return out


def aggregate_connectivity_by_ply(game_records: List[dict], ply_buckets) -> List[dict]:
    """Bucket per-position stats by (ply_bucket, color, outcome).

    `game_records` is a list of dicts each with: move_history, winner,
    active_size, start_player. Returns list of aggregate rows.
    """
    buckets: Dict = defaultdict(lambda: defaultdict(list))
    for gr in game_records:
        move_history = [(int(m["row"]), int(m["col"])) for m in (gr.get("moves") or [])]
        active = (gr.get("meta") or {}).get("board_size", 24)
        start_player = gr.get("starting_player") or (gr.get("meta") or {}).get("starting_player", "red")
        winner = gr.get("winner", "draw")
        state = TwixtState(active_size=active, to_move=start_player)

        for ply, (r, c) in enumerate(move_history):
            state = state.apply_move((r, c))
            stats = compute_position_connectivity(state)

            # Find matching ply bucket
            bucket_label = "other"
            for lo, hi, label in ply_buckets:
                if lo <= ply + 1 <= hi:
                    bucket_label = label
                    break

            key = (bucket_label, winner)
            buckets[key]["red_largest_component_size"].append(stats["red_largest_component_size"])
            buckets[key]["black_largest_component_size"].append(stats["black_largest_component_size"])
            buckets[key]["red_has_top_component"].append(int(stats["red_has_top_component"]))
            buckets[key]["red_has_bottom_component"].append(int(stats["red_has_bottom_component"]))
            buckets[key]["black_has_left_component"].append(int(stats["black_has_left_component"]))
            buckets[key]["black_has_right_component"].append(int(stats["black_has_right_component"]))
            buckets[key]["red_n_goal_touching_components"].append(stats["red_n_goal_touching_components"])
            buckets[key]["black_n_goal_touching_components"].append(stats["black_n_goal_touching_components"])

    rows = []
    for (bucket_label, outcome), data in sorted(buckets.items()):
        if not data.get("red_largest_component_size"):
            continue
        n = len(data["red_largest_component_size"])
        row = {"ply_bucket": bucket_label, "outcome": outcome, "n": n}
        for k, vs in data.items():
            row[f"mean_{k}"] = round(sum(vs) / n, 3)
        rows.append(row)
    return rows
