"""Connectivity-aware replay diagnostics — Phase 1 of the retrain design spec.

Computes per-position Twixt-structural stats (goal-touching components,
largest component size, etc.) from game JSON move histories, then aggregates
by ply bucket + outcome for analyzer-side reporting.
"""
from __future__ import annotations
import dataclasses
from typing import Dict, List, Optional, Tuple, FrozenSet, Set
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


def _apply_hypothetical(state: TwixtState, player: str, move: Tuple[int, int]) -> TwixtState:
    """Apply `move` as if it were `player`'s turn, returning the new state.

    Uses dataclasses.replace to swap to_move so apply_move's validation
    accepts the placement; otherwise mirrors engine semantics exactly.
    Raises ValueError if the move is not legal for `player`.
    """
    swapped = dataclasses.replace(state, to_move=player)
    return swapped.apply_move(move)


_RED_GOAL_KEYS = ("top", "bottom")
_BLACK_GOAL_KEYS = ("left", "right")


def _is_on_goal_side(player: str, side: str, r: int, c: int, active_size: int) -> bool:
    if player == "red":
        return (side == "top" and r == 0) or (side == "bottom" and r == active_size - 1)
    return (side == "left" and c == 0) or (side == "right" and c == active_size - 1)


def component_goal_distances(
    state: TwixtState,
    player: str,
    component: FrozenSet[Tuple[int, int]],
    max_depth: int = 3,
) -> dict:
    """Shortest fresh-placement distance from `component` to each goal side.

    For red: returns {"top", "bottom"} -> int in [0, max_depth] or None.
    For black: returns {"left", "right"}.

    Algorithm (spec 2026-05-03 §6.3): BFS where layer-0 frontier is the
    component's pegs (cost 0). Each transition to a new fresh placement
    (r, c) costs +1 and is gated by:
      - state.is_valid_placement(r, c)
      - _apply_hypothetical(state, player, (r, c)) succeeds (engine accepts
        legality + bridge crossing checks)
      - the new peg is in the same connected component as some frontier
        peg in the resulting state (which transitively absorbs same-color
        pegs the new bridges connected to)
    Stop when any frontier cell IS on the target goal side. Return None
    when no path within max_depth.
    """
    if player not in ("red", "black"):
        raise ValueError(f"Unknown player {player!r}")
    keys = _RED_GOAL_KEYS if player == "red" else _BLACK_GOAL_KEYS
    active = state.active_size
    out: Dict[str, Optional[int]] = {k: None for k in keys}

    # Distance-0: any peg in the component already on a goal side.
    for (r, c) in component:
        for side in keys:
            if _is_on_goal_side(player, side, r, c, active):
                out[side] = 0

    if all(out[k] == 0 for k in keys):
        return out

    for side in keys:
        if out[side] == 0:
            continue
        out[side] = _bfs_distance_to_goal(
            state, player, side, component, max_depth, active
        )
    return out


def _knight_neighbors(r: int, c: int):
    KNIGHT = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
    for dr, dc in KNIGHT:
        yield r + dr, c + dc


_BFS_MAX_NODES_EXPANDED = 5000  # guardrail per spec review; sub-millisecond on typical positions


def _bfs_distance_to_goal(
    state: TwixtState,
    player: str,
    side: str,
    component: FrozenSet[Tuple[int, int]],
    max_depth: int,
    active: int,
) -> Optional[int]:
    """One-side BFS over fresh placements.

    Bounded by both max_depth (logical layers) AND _BFS_MAX_NODES_EXPANDED
    (defensive guardrail against pathological positions). If the node cap
    is hit, returns None to signal "not reachable within budget" rather
    than raising.
    """
    visited_states: Set[FrozenSet[Tuple[int, int]]] = set()

    def _player_pegs(s: TwixtState) -> FrozenSet[Tuple[int, int]]:
        return frozenset(p for p, col in s.pegs.items() if col == player)

    visited_states.add(_player_pegs(state))

    layer = 0
    layer_states: List[Tuple[TwixtState, FrozenSet[Tuple[int, int]]]] = [(state, component)]
    nodes_expanded = 0

    while layer < max_depth:
        layer += 1
        next_layer: List[Tuple[TwixtState, FrozenSet[Tuple[int, int]]]] = []
        for (cur_state, cur_comp) in layer_states:
            # Use a player-perspective view for legality checks; cur_state.to_move
            # may not match `player` after one or more hypothetical placements.
            cur_state_as_player = (
                cur_state if cur_state.to_move == player
                else dataclasses.replace(cur_state, to_move=player)
            )
            candidates = set()
            for (r, c) in cur_comp:
                for nr, nc in _knight_neighbors(r, c):
                    if 0 <= nr < active and 0 <= nc < active:
                        candidates.add((nr, nc))
            for (nr, nc) in candidates:
                if (nr, nc) in cur_state.pegs:
                    continue
                if not cur_state_as_player.is_valid_placement(nr, nc):
                    continue
                try:
                    new_state = _apply_hypothetical(cur_state, player, (nr, nc))
                except (ValueError, AssertionError):
                    continue
                key = _player_pegs(new_state)
                if key in visited_states:
                    continue
                visited_states.add(key)
                nodes_expanded += 1
                if nodes_expanded > _BFS_MAX_NODES_EXPANDED:
                    return None
                new_comp = frozenset(new_state._get_connected_component((nr, nc), player))
                # Reject placements that fail to extend the current frontier:
                # the new peg's resulting component must overlap with cur_comp.
                # Without this guard, BFS could "teleport" through disjoint
                # components (e.g., a fresh peg whose bridge to cur_comp was
                # blocked by a crossing but happens to land on the goal line).
                if cur_comp.isdisjoint(new_comp):
                    continue
                if any(_is_on_goal_side(player, side, r, c, active) for (r, c) in new_comp):
                    return layer
                next_layer.append((new_state, new_comp))
        layer_states = next_layer
        if not layer_states:
            break
    return None
