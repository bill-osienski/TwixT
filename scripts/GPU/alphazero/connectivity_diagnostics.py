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


def _categorize_total(total: Optional[int], endpoint_distances: dict, max_depth: int) -> str:
    """Map a goal-completion state to one of six category strings (spec §6.2)."""
    if total is None:
        return "not_reachable"
    if total == 0:
        return "already_won"
    if total == 1:
        return "one_move_win"
    vals = sorted(v for v in endpoint_distances.values() if v is not None)
    if len(vals) == 2 and vals == [1, 1]:
        return "two_endpoint_closeout_2ply"
    if vals == [0, 2]:
        return "one_endpoint_distance_2"
    if total <= max_depth:
        return "broader_conversion"
    return "not_reachable"


def compute_goal_completion_state(
    state: TwixtState,
    player: str,
    max_depth: int = 3,
    min_component_size: int = 8,
    enumerate_moves: bool = True,
) -> Optional[dict]:
    """Best dominant-unclosed component for `player`, or None.

    Selection rule (spec §6.1): smallest total_goal_distance; tie-break by
    largest component size; tie-break by deterministic peg ordering (min-corner).

    Performance:
    `_enumerate_classification_moves` is the dominant cost — for each of
    ~30-50 candidate moves it spawns a hypothetical state and runs
    component_goal_distances at depth 3 (which itself iterates candidates
    and recurses). Per-ply analyzer walks that only need
    `total_goal_distance` and `category` should pass `enumerate_moves=False`
    to skip this. Watch-window classification (which needs
    `endpoint_completion_moves` to identify completion moves) and Phase 3
    inline capture leave the default True.

    When enumerate_moves=False the returned dict still has the same keys,
    but `endpoint_completion_moves` and `distance_reducing_moves` are
    empty lists. The `moves_enumerated` field distinguishes "no moves
    available" from "we skipped enumeration for speed."
    """
    pegs_of = [(r, c) for (r, c), col in state.pegs.items() if col == player]
    seen: Set[Tuple[int, int]] = set()
    components = []
    for peg in pegs_of:
        if peg in seen:
            continue
        comp = frozenset(state._get_connected_component(peg, player))
        seen.update(comp)
        if len(comp) >= min_component_size:
            components.append(comp)
    if not components:
        return None

    best = None
    best_key = None
    for comp in components:
        ed = component_goal_distances(state, player, comp, max_depth=max_depth)
        if any(v is None for v in ed.values()):
            total = None
        else:
            total = sum(ed.values())
        sort_total = total if total is not None else 10**9
        size = len(comp)
        min_corner = min(comp)
        key = (sort_total, -size, min_corner)
        if best_key is None or key < best_key:
            best_key = key
            best = (comp, ed, total)

    if best is None:
        return None
    comp, ed, total = best
    category = _categorize_total(total, ed, max_depth)
    if category == "not_reachable":
        return None

    completion_moves: list = []
    reducing_moves: list = []
    if total is not None and enumerate_moves:
        completion_moves, reducing_moves = _enumerate_classification_moves(
            state, player, comp, ed, total, max_depth
        )

    keys = _RED_GOAL_KEYS if player == "red" else _BLACK_GOAL_KEYS
    touches_a = ed[keys[0]] == 0
    touches_b = ed[keys[1]] == 0

    return {
        "component_pegs": comp,
        "largest_component_size": len(comp),
        "endpoint_distances": ed,
        "total_goal_distance": total,
        "touches_goal_a": touches_a,
        "touches_goal_b": touches_b,
        "endpoint_completion_moves": completion_moves,
        "distance_reducing_moves": reducing_moves,
        "moves_enumerated": enumerate_moves,
        "category": category,
        "max_depth": max_depth,
    }


def _enumerate_classification_moves(
    state: TwixtState,
    player: str,
    component: FrozenSet[Tuple[int, int]],
    endpoint_distances: dict,
    total_before: int,
    max_depth: int,
) -> Tuple[list, list]:
    """Return (endpoint_completion_moves, distance_reducing_moves) for a component.

    - distance_reducing_moves: fresh placements that strictly reduce total_goal_distance.
    - endpoint_completion_moves: subset that drops a non-zero endpoint distance to 0.
      (Spec §6.1 prose definition. NOT "wins the game in one move" — that would
      falsely exclude moves that close one endpoint while the other remains.)

    Apply player-perspective legality + disjoint-component guards.
    """
    active = state.active_size

    candidates = set()
    for (r, c) in component:
        for nr, nc in _knight_neighbors(r, c):
            if 0 <= nr < active and 0 <= nc < active:
                candidates.add((nr, nc))

    state_as_player = (
        state if state.to_move == player
        else dataclasses.replace(state, to_move=player)
    )

    completion: list = []
    reducing: list = []
    for cand in sorted(candidates):
        if cand in state.pegs:
            continue
        if not state_as_player.is_valid_placement(*cand):
            continue
        try:
            new_state = _apply_hypothetical(state, player, cand)
        except (ValueError, AssertionError):
            continue
        new_comp = frozenset(new_state._get_connected_component(cand, player))
        # Disjoint-component guard: candidate must extend the dominant component.
        if component.isdisjoint(new_comp):
            continue
        new_ed = component_goal_distances(new_state, player, new_comp, max_depth=max_depth)
        if any(v is None for v in new_ed.values()):
            continue
        new_total = sum(new_ed.values())
        if new_total < total_before:
            reducing.append(cand)
            # Spec §6.1: completion = drops a non-zero endpoint to 0.
            # (NOT "wins the game in one move" — the latter would falsely exclude
            # moves that close one endpoint while the other remains at distance 1.)
            if any((prev > 0) and (new_ed[k] == 0) for k, prev in endpoint_distances.items()):
                completion.append(cand)
    return completion, reducing


def classify_selected_conversion_move(
    state_before: TwixtState,
    player: str,
    selected_move: Tuple[int, int],
    goal_state_before: dict,
    max_depth: int = 3,
    min_component_size: int = 8,
) -> dict:
    """Classify a selected move against the pre-move dominant-unclosed state.

    Raw booleans are non-exclusive; primary_class is priority-resolved
    (completes_endpoint > reduces_total_goal_distance > redundant_reinforcement
    > off_chain > other) — used for report rate-summing.
    """
    component = goal_state_before["component_pegs"]
    total_before = goal_state_before.get("total_goal_distance")
    completion_moves = set(map(tuple, goal_state_before.get("endpoint_completion_moves") or []))
    reducing_moves = set(map(tuple, goal_state_before.get("distance_reducing_moves") or []))

    selected = tuple(selected_move)
    completes = selected in completion_moves
    reduces = selected in reducing_moves

    try:
        new_state = _apply_hypothetical(state_before, player, selected)
    except (ValueError, AssertionError):
        return {
            "completes_endpoint": False,
            "reduces_total_goal_distance": False,
            "is_redundant_reinforcement": False,
            "is_off_chain": False,
            "primary_class": "other",
            "total_goal_distance_before": total_before,
            "total_goal_distance_after": None,
        }
    new_comp = frozenset(new_state._get_connected_component(selected, player))
    new_ed = component_goal_distances(new_state, player, new_comp, max_depth=max_depth)
    if any(v is None for v in new_ed.values()):
        total_after = None
    else:
        total_after = sum(new_ed.values())

    bridgeable_to_component = bool(new_comp & component)
    if total_before is not None and total_after is not None:
        no_reduction = (total_after >= total_before)
    else:
        no_reduction = (total_after is None)
    redundant = bridgeable_to_component and no_reduction and not reduces

    has_knight_neighbor_in_component = any(
        (selected[0] + dr, selected[1] + dc) in component
        for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
    )
    off_chain = (not has_knight_neighbor_in_component) and (not reduces)

    if completes:
        primary = "completes_endpoint"
    elif reduces:
        primary = "reduces_total_goal_distance"
    elif redundant:
        primary = "redundant_reinforcement"
    elif off_chain:
        primary = "off_chain"
    else:
        primary = "other"

    return {
        "completes_endpoint": completes,
        "reduces_total_goal_distance": reduces,
        "is_redundant_reinforcement": redundant,
        "is_off_chain": off_chain,
        "primary_class": primary,
        "total_goal_distance_before": total_before,
        "total_goal_distance_after": total_after,
    }
