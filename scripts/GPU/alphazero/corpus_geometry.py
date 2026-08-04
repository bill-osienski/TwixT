"""Atlas corpus geometry -- design section 3, EXECUTION-FROZEN.

Pure and reservoir-free: everything here is decidable from non-search-derived
game metadata, which is what lets Stage 2 qualify without generating anything.

CPU-SAFE: stdlib only, no MLX, no scipy.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PHASES: Tuple[str, ...] = ("opening", "early_mid", "midgame", "late")
SIDES: Tuple[str, ...] = ("red", "black")
SPLITS: Tuple[str, ...] = ("discovery", "validation")

# (name, first_ply, last_ply_inclusive_or_None)
_PHASE_BOUNDS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("opening", 0, 30),
    ("early_mid", 31, 60),
    ("midgame", 61, 90),
    ("late", 91, None),
)

ALLOWED_N = (200, 240, 280, 320, 360, 400)
MAX_SEED_RANGE_GAMES = 480
PILOT_GAMES = 24
PILOT_PER_CELL = 3
SIZING_MARGIN = 1.20


def phase_for_ply(ply: int) -> str:
    if ply < 0:
        raise ValueError(f"ply must be non-negative, got {ply}")
    for name, lo, hi in _PHASE_BOUNDS:
        if ply >= lo and (hi is None or ply <= hi):
            return name
    raise AssertionError(f"unreachable: no phase for ply {ply}")


def side_for_ply(ply: int, start_player: str) -> str:
    if start_player not in SIDES:
        raise ValueError(f"start_player must be one of {SIDES}, got {start_player!r}")
    if ply % 2 == 0:
        return start_player
    return "black" if start_player == "red" else "red"


@dataclass(frozen=True)
class GameMeta:
    """Non-search-derived game metadata. Nothing here reads a search result."""
    game_id: int
    seed: int
    n_moves: int
    start_player: str


def eligible_plies(meta: GameMeta, phase: str, side: str) -> List[int]:
    return [
        p for p in range(meta.n_moves)
        if phase_for_ply(p) == phase and side_for_ply(p, meta.start_player) == side
    ]


def eligible_cells(meta: GameMeta) -> Set[Tuple[str, str]]:
    """The (phase, side) cells this game can serve at all."""
    return {
        (phase_for_ply(p), side_for_ply(p, meta.start_player))
        for p in range(meta.n_moves)
    }


def stable_key(sampling_seed: int, game_id: int, split: str, phase: str,
               side: str, ply: int) -> str:
    """SHA-1 over the frozen tuple. NEVER Python hash(): it is
    process-randomized for str/bytes, so a rerun would silently reorder ties
    and produce a different corpus.
    """
    raw = f"{sampling_seed}|{game_id}|{split}|{phase}|{side}|{ply}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    assignment: Dict[int, Tuple[str, str, str]]      # game_id -> (split, phase, side)
    achieved_flow: int
    demanded_flow: int
    unmet: Dict[Tuple[str, str, str], int]
    min_cut_games: List[int]
    min_cut_cells: List[Tuple[str, str, str]]

    @property
    def complete(self) -> bool:
        return self.achieved_flow == self.demanded_flow


def match_games_to_cells(games: Sequence[GameMeta],
                         demands: Dict[Tuple[str, str, str], int],
                         sampling_seed: int) -> MatchResult:
    """Deterministic bipartite b-matching by Edmonds-Karp.

    Games have capacity ONE (design section 3's one-position-per-game rule), so
    a game that could serve several cells still supplies a single row -- the
    shared-capacity effect that per-cell capacity checks miss entirely.

    Determinism: adjacency is built in stable-digest order, so the augmenting
    search visits candidates identically on every run and no Python hash()
    ordering leaks in.
    """
    cells = sorted(demands)
    n_g = len(games)
    src, sink = 0, n_g + len(cells) + 1
    cell_node = {c: n_g + 1 + i for i, c in enumerate(cells)}
    cap: Dict[int, Dict[int, int]] = {i: {} for i in range(sink + 1)}

    def add(u: int, v: int, c: int) -> None:
        cap[u][v] = cap[u].get(v, 0) + c
        cap[v].setdefault(u, 0)

    ordered_games = sorted(
        games,
        key=lambda g: stable_key(sampling_seed, g.game_id, "-", "-", "-", 0),
    )
    for gi, g in enumerate(ordered_games, start=1):
        add(src, gi, 1)
        elig = eligible_cells(g)
        for c in cells:
            _split, phase, side = c
            if (phase, side) in elig:
                add(gi, cell_node[c], 1)
    for c in cells:
        add(cell_node[c], sink, demands[c])

    flow = 0
    while True:
        parent: Dict[int, Optional[int]] = {src: None}
        q = deque([src])
        while q and sink not in parent:
            u = q.popleft()
            for v, c in cap[u].items():
                if c > 0 and v not in parent:
                    parent[v] = u
                    q.append(v)
        if sink not in parent:
            break
        path, v = [], sink
        while v != src:
            u = parent[v]
            path.append((u, v))
            v = u
        bottleneck = min(cap[u][v] for u, v in path)
        for u, v in path:
            cap[u][v] -= bottleneck
            cap[v][u] += bottleneck
        flow += bottleneck

    # residual reachability from the source == the min cut's source side
    seen, q = {src}, deque([src])
    while q:
        u = q.popleft()
        for v, c in cap[u].items():
            if c > 0 and v not in seen:
                seen.add(v)
                q.append(v)

    assignment: Dict[int, Tuple[str, str, str]] = {}
    for gi, g in enumerate(ordered_games, start=1):
        for c in cells:
            if cap[cell_node[c]].get(gi, 0) > 0:      # reverse edge carries flow
                assignment[g.game_id] = c
                break

    unmet: Dict[Tuple[str, str, str], int] = {}
    for c in cells:
        shortfall = cap[cell_node[c]].get(sink, 0)   # residual == unserved demand
        if shortfall:
            unmet[c] = shortfall
    return MatchResult(
        assignment=assignment,
        achieved_flow=flow,
        demanded_flow=sum(demands.values()),
        unmet=unmet,
        min_cut_games=[g.game_id for gi, g in enumerate(ordered_games, start=1)
                       if gi in seen],
        min_cut_cells=[c for c in cells if cell_node[c] not in seen],
    )


# ---------------------------------------------------------------------------
# Pilot gate and sizing
# ---------------------------------------------------------------------------

def pilot_geometry_gate(pilot_games: Sequence[GameMeta],
                        sampling_seed: int) -> Dict[str, object]:
    """Design section 3: the first 24 games must admit a complete matching of
    three positions into each of the eight phase/side cells.

    Runs BEFORE the pilot ladder. A no-go here costs nothing but generation of
    the pilot block, which is the entire point.
    """
    if len(pilot_games) != PILOT_GAMES:
        raise ValueError(
            f"pilot must be exactly {PILOT_GAMES} games, got {len(pilot_games)}")
    demands = {("discovery", p, s): PILOT_PER_CELL for p in PHASES for s in SIDES}
    r = match_games_to_cells(pilot_games, demands, sampling_seed)
    return {
        "verdict": "PASS" if r.complete else "PHASE_GEOMETRY_NO_GO",
        "assignment": r.assignment,
        "achieved_flow": r.achieved_flow,
        "demanded_flow": r.demanded_flow,
        "unmet": r.unmet,
        "min_cut_cells": r.min_cut_cells,
    }


def _round_up_to_multiple_of_40(x: int) -> int:
    return ((x + 39) // 40) * 40


def size_continuation(pilot_games: Sequence[GameMeta],
                      n_target: int) -> Dict[str, object]:
    """Design section 3's frozen sizing rule.

    Sweeps EVERY nonempty proper subset of the 8 phase/side cells (254 of them),
    because per-cell checks miss shared capacity: one late game can serve either
    side but supplies only one row.
    """
    if n_target not in ALLOWED_N:
        raise ValueError(f"n_target must be one of {ALLOWED_N}, got {n_target}")
    if len(pilot_games) != PILOT_GAMES:
        raise ValueError(f"pilot must be exactly {PILOT_GAMES} games")

    d_c = n_target // 8 - PILOT_PER_CELL
    cells = [(p, s) for p in PHASES for s in SIDES]
    pilot_cells = [eligible_cells(g) for g in pilot_games]

    best_g, binding, worst_q = 0, None, None
    for r in range(1, len(cells)):
        for subset in itertools.combinations(cells, r):
            sset = set(subset)
            servers = sum(1 for ec in pilot_cells if ec & sset)
            q_s = servers / PILOT_GAMES
            d_s = d_c * len(subset)
            if d_s == 0:
                continue
            if q_s == 0.0:
                return {
                    "verdict": "PHASE_GEOMETRY_NO_GO",
                    "reason": f"q_S = 0 for demanded subset {sorted(subset)}",
                    "binding_subset": sorted(subset), "q_S": 0.0,
                    "g_cont": None, "G_total": None,
                }
            g_s = math.ceil(SIZING_MARGIN * d_s / q_s)
            if g_s > best_g:
                best_g, binding, worst_q = g_s, sorted(subset), q_s

    g_cont = max(n_target - PILOT_GAMES, best_g)
    g_total = _round_up_to_multiple_of_40(PILOT_GAMES + g_cont)
    if g_total > MAX_SEED_RANGE_GAMES:
        return {
            "verdict": "PHASE_GEOMETRY_NO_GO",
            "reason": (f"G_total {g_total} exceeds the frozen "
                       f"{MAX_SEED_RANGE_GAMES}-game range"),
            "binding_subset": binding, "q_S": worst_q,
            "g_cont": g_cont, "G_total": g_total,
        }
    return {
        "verdict": "OK", "g_cont": g_cont, "G_total": g_total,
        "binding_subset": binding, "q_S": worst_q, "d_c": d_c,
    }


# ---------------------------------------------------------------------------
# Final assignment
# ---------------------------------------------------------------------------

def final_demands(n_target: int) -> Dict[Tuple[str, str, str], int]:
    """Per split x phase x side. discovery = 3N/40 - 3, validation = N/20.

    They sum to d_c = N/8 - 3 per phase/side cell and to N - 24 overall. Every
    allowed N is a multiple of 40, which is what makes all three integral.
    """
    if n_target not in ALLOWED_N:
        raise ValueError(f"n_target must be one of {ALLOWED_N}, got {n_target}")
    disc = 3 * n_target // 40 - PILOT_PER_CELL
    val = n_target // 20
    d: Dict[Tuple[str, str, str], int] = {}
    for p in PHASES:
        for s in SIDES:
            d[("discovery", p, s)] = disc
            d[("validation", p, s)] = val
    return d


def select_ply(meta: GameMeta, split: str, phase: str, side: str,
               sampling_seed: int) -> int:
    """Lowest stable digest among the cell's eligible plies."""
    plies = eligible_plies(meta, phase, side)
    if not plies:
        raise ValueError(f"game {meta.game_id} has no eligible ply for {phase}/{side}")
    return min(plies, key=lambda p: stable_key(
        sampling_seed, meta.game_id, split, phase, side, p))


def assign_corpus(pilot_assignment: Dict[int, Tuple[str, str, str]],
                  continuation_games: Sequence[GameMeta],
                  n_target: int, sampling_seed: int) -> Dict[str, object]:
    """Final matching. Pilot rows are FIXED and never reconsidered.

    On shortfall this STOPS and emits the failure artifact. It never tops up,
    rebalances cells, moves pilot rows, or relaxes one-position-per-game.
    """
    # NOTE: do NOT decrement demands for the pilot rows. `final_demands` ALREADY
    # nets them out -- discovery = 3N/40 - 3, where the -3 IS the pilot's three
    # rows per cell, and the demands therefore sum to N - 24, the continuation
    # count. Subtracting again would double-count the pilot.
    demands = final_demands(n_target)
    pilot_ids = set(pilot_assignment)
    pool = [g for g in continuation_games if g.game_id not in pilot_ids]

    r = match_games_to_cells(pool, demands, sampling_seed)
    if not r.complete:
        return {
            "verdict": "ASSIGNMENT_SHORTFALL",
            "demands": {"|".join(k): v for k, v in demands.items()},
            "raw_capacity": {
                "|".join(k): sum(1 for g in pool if (k[1], k[2]) in eligible_cells(g))
                for k in demands},
            "achieved_flow": r.achieved_flow,
            "demanded_flow": r.demanded_flow,
            "unmet": {"|".join(k): v for k, v in r.unmet.items()},
            "min_cut_cells": ["|".join(c) for c in r.min_cut_cells],
            "min_cut_games": r.min_cut_games,
        }

    by_id = {g.game_id: g for g in pool}
    rows = []
    for gid in sorted(r.assignment):
        split, phase, side = r.assignment[gid]
        rows.append({
            "game_id": gid, "seed": by_id[gid].seed, "split": split,
            "phase": phase, "side": side,
            "ply": select_ply(by_id[gid], split, phase, side, sampling_seed),
        })
    return {"verdict": "OK", "rows": rows,
            "achieved_flow": r.achieved_flow, "demanded_flow": r.demanded_flow}
