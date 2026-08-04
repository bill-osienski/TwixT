# Atlas Stage 2 — Corpus Generation, Assignment and Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and qualify the corpus generator, the deterministic bipartite assignment, the 254-subset geometry checks and the failure artifacts — stopping before any fresh-reservoir generation or GPU run.

**Architecture:** One pure module `corpus_geometry.py` holds everything decidable without a reservoir: phase bucketing, eligibility, stable-digest ordering, max-flow matching with a min-cut witness, the pilot geometry gate and the subset sizing formula. A thin `build_atlas_corpus.py` CLI emits the frozen generation protocol and the generation command **without running it**. The whole geometry layer is testable on synthetic metadata, because eligibility derives from `(n_moves, start_player)` alone.

**Tech Stack:** Python 3, stdlib only (`hashlib`, `math`, `itertools`, `json`, `argparse`, `collections`). No scipy. Tests: `.venv/bin/python -m pytest -p no:cacheprovider`.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-03-convergence-atlas-design.md` §3 — **§3–§12 are EXECUTION-FROZEN.** No parameter, threshold, predicate, producer or geometry rule may be changed.
- **STOP BEFORE GENERATION.** This stage builds and qualifies tooling. It **does not** generate the reservoir, does not run self-play, and does not touch a GPU. The CLI emits the generation command; running it is a separate authorization, as Phase 0's preflight was.
- **No `mcts.py` change.** The Stage 1 scoped exception covers diagnostic observer surfaces only, and Stage 2 needs none.
- **Eligibility uses only non-search-derived metadata:** `game_id`, `seed`, `n_moves`, `start_player`. Never a value, residual, entropy, branching count or outcome.
- **Frozen phase bounds:** opening `0–30`, early-mid `31–60`, midgame `61–90`, late `91+`.
- **Two cell spaces, do not conflate them.** Sizing and the pilot gate use the **8 phase×side** cells. The final matching uses the **16 split×phase×side** cells.
- **Frozen demands:** `d_c = N/8 − 3` residual per phase×side cell; final per-cell `discovery = 3N/40 − 3`, `validation = N/20`. They sum to `d_c`, and to `N − 24` overall.
- **`N` ∈ {200, 240, 280, 320, 360, 400}.** Seed range frozen at **480 games** maximum.
- **Stable digest ordering** derived only from `(sampling_seed, game identity, split, phase, side, ply)`. **Never Python `hash()`** — it is process-randomized for `str`/`bytes`, so a rerun would silently produce a different corpus.
- **One position per game.** Never top up, rebalance cells, move pilot rows, or relax one-position-per-game.
- **Zero denominators and undefined statistics are `None`, never `0` and never `false`.**
- Commit after every task.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/GPU/alphazero/corpus_geometry.py` (create) | Pure: phases, eligibility, stable ordering, max-flow matching + min-cut witness, pilot gate, subset sizing. |
| `scripts/GPU/alphazero/build_atlas_corpus.py` (create) | CLI: `emit-protocol`, `emit-gen-command`, `pilot-gate`, `size`, `assign`. Emits commands; runs none. |
| `tests/test_corpus_geometry.py` (create) | Phases, eligibility, ordering, matching, min-cut, pilot gate, sizing — all on synthetic metadata. |
| `tests/test_build_atlas_corpus_cli.py` (create) | Protocol emission, the operator stop, and failure-artifact shape. |

---

### Task 1: Phases, eligibility and stable ordering

**Files:**
- Create: `scripts/GPU/alphazero/corpus_geometry.py`
- Test: `tests/test_corpus_geometry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PHASES`, `SIDES`, `SPLITS`, `phase_for_ply(ply) -> str`, `side_for_ply(ply, start_player) -> str`, `@dataclass GameMeta(game_id, seed, n_moves, start_player)`, `eligible_plies(meta, phase, side) -> list[int]`, `eligible_cells(meta) -> set[tuple[str, str]]`, `stable_key(sampling_seed, game_id, split, phase, side, ply) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_corpus_geometry.py
import pytest

from scripts.GPU.alphazero.corpus_geometry import (
    GameMeta, PHASES, SIDES, eligible_cells, eligible_plies, phase_for_ply,
    side_for_ply, stable_key,
)


@pytest.mark.parametrize("ply,expected", [
    (0, "opening"), (30, "opening"), (31, "early_mid"), (60, "early_mid"),
    (61, "midgame"), (90, "midgame"), (91, "late"), (279, "late"),
])
def test_phase_bounds_are_exact(ply, expected):
    assert phase_for_ply(ply) == expected


def test_side_alternates_from_the_start_player():
    assert side_for_ply(0, "red") == "red"
    assert side_for_ply(1, "red") == "black"
    assert side_for_ply(0, "black") == "black"
    assert side_for_ply(91, "red") == "black"      # odd ply


def test_eligibility_derives_only_from_n_moves_and_start_player():
    """No value, residual, entropy, branching or outcome is consulted."""
    short = GameMeta(game_id=1, seed=100, n_moves=40, start_player="red")
    assert eligible_plies(short, "opening", "red") == list(range(0, 31, 2))
    assert eligible_plies(short, "late", "red") == []        # never reaches ply 91
    cells = eligible_cells(short)
    assert ("opening", "red") in cells and ("early_mid", "black") in cells
    assert not any(p == "late" for p, _s in cells)


def test_a_long_game_serves_every_cell():
    long_game = GameMeta(game_id=2, seed=101, n_moves=200, start_player="red")
    assert eligible_cells(long_game) == {(p, s) for p in PHASES for s in SIDES}


def test_stable_key_is_deterministic_across_processes():
    """Python hash() is process-randomized for str; a rerun would silently
    produce a different corpus. The key must be a digest."""
    a = stable_key(20260804, 7, "discovery", "late", "black", 95)
    b = stable_key(20260804, 7, "discovery", "late", "black", 95)
    assert a == b and len(a) == 40 and a != stable_key(20260804, 7, "discovery", "late", "black", 97)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_corpus_geometry.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named '...corpus_geometry'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/corpus_geometry.py
"""Atlas corpus geometry -- design section 3, EXECUTION-FROZEN.

Pure and reservoir-free: everything here is decidable from non-search-derived
game metadata, which is what lets Stage 2 qualify without generating anything.

CPU-SAFE: stdlib only, no MLX, no scipy.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_corpus_geometry.py -v -p no:cacheprovider`
Expected: PASS — 12 passed (the phase test is parametrized over 8 cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/corpus_geometry.py tests/test_corpus_geometry.py
git commit -m "feat(atlas-s2): phases, eligibility and stable digest ordering"
```

---

### Task 2: Max-flow matching with a min-cut witness

**Files:**
- Modify: `scripts/GPU/alphazero/corpus_geometry.py`
- Test: `tests/test_corpus_geometry.py`

**Interfaces:**
- Consumes: `GameMeta`, `eligible_cells` from Task 1.
- Produces: `match_games_to_cells(games, demands, sampling_seed) -> MatchResult` with `MatchResult(assignment: dict[int, tuple], achieved_flow: int, demanded_flow: int, unmet: dict[tuple, int], min_cut_games: list[int], min_cut_cells: list[tuple])`.

Edmonds–Karp on a small graph: source → game (cap 1) → cell (cap 1 per eligible edge) → sink (cap = demand). At most 480 games and 16 cells, so at most ~376 augmentations over ~8k edges — trivially fast, stdlib only, and the min cut falls out of residual reachability.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_corpus_geometry.py
from scripts.GPU.alphazero.corpus_geometry import match_games_to_cells


def _games(specs):
    return [GameMeta(game_id=i, seed=1000 + i, n_moves=n, start_player=sp)
            for i, (n, sp) in enumerate(specs)]


def test_full_matching_succeeds_and_respects_one_position_per_game():
    games = _games([(200, "red")] * 4)
    demands = {("discovery", "late", "red"): 2, ("discovery", "late", "black"): 2}
    r = match_games_to_cells(games, demands, sampling_seed=1)
    assert r.achieved_flow == r.demanded_flow == 4
    assert len(r.assignment) == 4               # one cell per game, no reuse
    assert len(set(r.assignment)) == 4
    assert not r.unmet


def test_shortfall_reports_unmet_cells_and_a_min_cut():
    """Four short games cannot serve a late cell at all."""
    games = _games([(40, "red")] * 4)
    demands = {("discovery", "late", "red"): 2}
    r = match_games_to_cells(games, demands, sampling_seed=1)
    assert r.achieved_flow == 0 and r.demanded_flow == 2
    assert r.unmet == {("discovery", "late", "red"): 2}
    assert ("discovery", "late", "red") in r.min_cut_cells


def test_shared_capacity_is_caught_not_double_counted():
    """One long game can serve either late side but supplies only ONE row."""
    games = _games([(200, "red")])
    demands = {("discovery", "late", "red"): 1, ("discovery", "late", "black"): 1}
    r = match_games_to_cells(games, demands, sampling_seed=1)
    assert r.achieved_flow == 1 and r.demanded_flow == 2
    assert sum(r.unmet.values()) == 1


def test_matching_is_deterministic_under_the_sampling_seed():
    games = _games([(200, "red")] * 6)
    demands = {("discovery", "late", "red"): 3}
    a = match_games_to_cells(games, demands, sampling_seed=42).assignment
    b = match_games_to_cells(games, demands, sampling_seed=42).assignment
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_corpus_geometry.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'match_games_to_cells'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/GPU/alphazero/corpus_geometry.py
from collections import deque


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
    game_index = {g.game_id: g for g in games}
    # node ids: 0 = source, 1..G = games, G+1.. = cells, last = sink
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
        parent = {src: None}
        q = deque([src])
        while q and sink not in parent:
            u = q.popleft()
            for v, c in cap[u].items():
                if c > 0 and v not in parent:
                    parent[v] = u
                    q.append(v)
        if sink not in parent:
            break
        # unit augmentation: every source and cell edge is integral
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

    demanded = sum(demands.values())
    unmet: Dict[Tuple[str, str, str], int] = {}
    for c in cells:
        served = cap[cell_node[c]].get(sink, 0)
        shortfall = served  # residual capacity to sink == unserved demand
        if shortfall:
            unmet[c] = shortfall
    return MatchResult(
        assignment=assignment,
        achieved_flow=flow,
        demanded_flow=demanded,
        unmet=unmet,
        min_cut_games=[g.game_id for gi, g in enumerate(ordered_games, start=1)
                       if gi in seen],
        min_cut_cells=[c for c in cells if cell_node[c] not in seen],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_corpus_geometry.py -v -p no:cacheprovider`
Expected: PASS — 16 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/corpus_geometry.py tests/test_corpus_geometry.py
git commit -m "feat(atlas-s2): deterministic max-flow matching with min-cut witness"
```

---

### Task 3: The pilot geometry gate

**Files:**
- Modify: `scripts/GPU/alphazero/corpus_geometry.py`
- Test: `tests/test_corpus_geometry.py`

**Interfaces:**
- Consumes: `match_games_to_cells`.
- Produces: `pilot_geometry_gate(pilot_games, sampling_seed) -> dict` with keys `verdict` (`"PASS"` | `"PHASE_GEOMETRY_NO_GO"`), `assignment`, `unmet`, `min_cut_cells`.

24 games, 3 into each of the 8 phase×side cells, **all discovery**. Fires **before** the pilot ladder — the whole point is to spend nothing if the geometry is infeasible.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_corpus_geometry.py
from scripts.GPU.alphazero.corpus_geometry import PILOT_GAMES, pilot_geometry_gate


def test_pilot_gate_passes_when_24_long_games_cover_every_cell():
    games = _games([(200, "red" if i % 2 == 0 else "black")
                    for i in range(PILOT_GAMES)])
    r = pilot_geometry_gate(games, sampling_seed=7)
    assert r["verdict"] == "PASS"
    assert len(r["assignment"]) == PILOT_GAMES


def test_pilot_gate_no_gos_when_late_cells_cannot_be_filled():
    """All games end before ply 91, so the two late cells are unfillable."""
    games = _games([(60, "red")] * PILOT_GAMES)
    r = pilot_geometry_gate(games, sampling_seed=7)
    assert r["verdict"] == "PHASE_GEOMETRY_NO_GO"
    assert ("discovery", "late", "red") in r["min_cut_cells"]


def test_pilot_gate_rejects_a_wrong_sized_pilot():
    with pytest.raises(ValueError):
        pilot_geometry_gate(_games([(200, "red")] * 23), sampling_seed=7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_corpus_geometry.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'pilot_geometry_gate'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/GPU/alphazero/corpus_geometry.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_corpus_geometry.py -v -p no:cacheprovider`
Expected: PASS — 19 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/corpus_geometry.py tests/test_corpus_geometry.py
git commit -m "feat(atlas-s2): pilot geometry gate, 3 per cell across 8 phase/side cells"
```

---

### Task 4: The 254-subset sizing formula

**Files:**
- Modify: `scripts/GPU/alphazero/corpus_geometry.py`
- Test: `tests/test_corpus_geometry.py`

**Interfaces:**
- Consumes: `eligible_cells`, `ALLOWED_N`, `SIZING_MARGIN`, `MAX_SEED_RANGE_GAMES`.
- Produces: `size_continuation(pilot_games, n_target) -> dict` with `g_cont`, `G_total`, `binding_subset`, `q_S`, `verdict`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_corpus_geometry.py
import itertools

from scripts.GPU.alphazero.corpus_geometry import (
    MAX_SEED_RANGE_GAMES, size_continuation,
)


def test_subset_sweep_covers_exactly_254_proper_nonempty_subsets():
    cells = [(p, s) for p in PHASES for s in SIDES]
    assert len(cells) == 8
    subsets = [c for r in range(1, 8) for c in itertools.combinations(cells, r)]
    assert len(subsets) == 2 ** 8 - 2 == 254


def test_sizing_never_exceeds_480_when_the_pilot_gate_passed():
    """q_S >= k/8 follows from the gate (3 distinct games per cell), so
    D_S/q_S <= N-24 and G_total <= 480 at the maximum N. Both stops are then
    invariant assertions, not live gates."""
    games = _games([(200, "red" if i % 2 == 0 else "black")
                    for i in range(PILOT_GAMES)])
    for n in (200, 240, 320, 400):
        r = size_continuation(games, n_target=n)
        assert r["verdict"] == "OK"
        assert r["G_total"] <= MAX_SEED_RANGE_GAMES
        assert r["G_total"] % 40 == 0
        assert r["g_cont"] >= n - PILOT_GAMES


def test_sizing_reports_the_binding_subset():
    games = _games([(200, "red" if i % 2 == 0 else "black")
                    for i in range(PILOT_GAMES)])
    r = size_continuation(games, n_target=240)
    assert r["binding_subset"] is not None
    assert 1 <= len(r["binding_subset"]) <= 7


def test_sizing_rejects_a_disallowed_n():
    games = _games([(200, "red")] * PILOT_GAMES)
    with pytest.raises(ValueError):
        size_continuation(games, n_target=250)


def test_zero_capacity_subset_is_a_no_go():
    """A pilot whose games never reach late leaves q_S = 0 for late-only subsets."""
    games = _games([(60, "red" if i % 2 == 0 else "black")
                    for i in range(PILOT_GAMES)])
    r = size_continuation(games, n_target=200)
    assert r["verdict"] == "PHASE_GEOMETRY_NO_GO"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_corpus_geometry.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'size_continuation'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/GPU/alphazero/corpus_geometry.py
import itertools


def _round_up_to_multiple_of_40(x: int) -> int:
    return ((x + 39) // 40) * 40


def size_continuation(pilot_games: Sequence[GameMeta], n_target: int) -> Dict[str, object]:
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
            "reason": f"G_total {g_total} exceeds the frozen {MAX_SEED_RANGE_GAMES}-game range",
            "binding_subset": binding, "q_S": worst_q,
            "g_cont": g_cont, "G_total": g_total,
        }
    return {
        "verdict": "OK", "g_cont": g_cont, "G_total": g_total,
        "binding_subset": binding, "q_S": worst_q, "d_c": d_c,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_corpus_geometry.py -v -p no:cacheprovider`
Expected: PASS — 24 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/corpus_geometry.py tests/test_corpus_geometry.py
git commit -m "feat(atlas-s2): 254-subset sizing with binding-subset reporting"
```

---

### Task 5: Final assignment and the failure artifact

**Files:**
- Modify: `scripts/GPU/alphazero/corpus_geometry.py`
- Test: `tests/test_corpus_geometry.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `final_demands(n_target) -> dict`, `assign_corpus(pilot_assignment, continuation_games, n_target, sampling_seed) -> dict` emitting the witness or the failure artifact, and `select_ply(meta, split, phase, side, sampling_seed) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_corpus_geometry.py
from scripts.GPU.alphazero.corpus_geometry import (
    assign_corpus, final_demands, select_ply,
)


@pytest.mark.parametrize("n,disc,val", [
    (200, 12, 10), (240, 15, 12), (280, 18, 14),
    (320, 21, 16), (360, 24, 18), (400, 27, 20),
])
def test_final_demands_are_integral_and_sum_to_d_c(n, disc, val):
    d = final_demands(n)
    assert d[("discovery", "opening", "red")] == disc
    assert d[("validation", "opening", "red")] == val
    assert disc + val == n // 8 - 3
    assert sum(d.values()) == n - 24


def test_select_ply_is_stable_and_within_the_cell():
    meta = GameMeta(game_id=5, seed=9, n_moves=200, start_player="red")
    a = select_ply(meta, "discovery", "late", "black", sampling_seed=3)
    assert a == select_ply(meta, "discovery", "late", "black", sampling_seed=3)
    assert phase_for_ply(a) == "late" and side_for_ply(a, "red") == "black"


def test_assign_corpus_emits_a_witness_on_success():
    pilot = _games([(200, "red" if i % 2 == 0 else "black") for i in range(PILOT_GAMES)])
    pa = pilot_geometry_gate(pilot, sampling_seed=7)["assignment"]
    cont = [GameMeta(game_id=100 + i, seed=2000 + i, n_moves=200,
                     start_player="red" if i % 2 == 0 else "black")
            for i in range(400)]
    r = assign_corpus(pa, cont, n_target=200, sampling_seed=7)
    assert r["verdict"] == "OK"
    assert len(r["rows"]) == 200 - PILOT_GAMES
    assert len({row["game_id"] for row in r["rows"]}) == len(r["rows"])


def test_assign_corpus_stops_with_a_failure_artifact_on_shortfall():
    """Continuation games that never reach late cannot fill the late cells, and
    the artifact must name the binding cut rather than silently topping up."""
    pilot = _games([(200, "red" if i % 2 == 0 else "black") for i in range(PILOT_GAMES)])
    pa = pilot_geometry_gate(pilot, sampling_seed=7)["assignment"]
    cont = [GameMeta(game_id=100 + i, seed=2000 + i, n_moves=50,
                     start_player="red" if i % 2 == 0 else "black")
            for i in range(400)]
    r = assign_corpus(pa, cont, n_target=200, sampling_seed=7)
    assert r["verdict"] == "ASSIGNMENT_SHORTFALL"
    assert r["achieved_flow"] < r["demanded_flow"]
    assert r["unmet"] and r["min_cut_cells"]
    assert "rows" not in r          # no partial corpus is emitted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_corpus_geometry.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'assign_corpus'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/GPU/alphazero/corpus_geometry.py
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
    # NOTE: do NOT decrement demands for the pilot rows. `final_demands`
    # ALREADY nets them out -- discovery = 3N/40 - 3, where the -3 IS the pilot's
    # three rows per cell, and the demands therefore sum to N - 24, the
    # continuation count. Subtracting again would double-count the pilot and
    # under-demand every discovery cell by three.
    demands = final_demands(n_target)
    pilot_ids = set(pilot_assignment)
    pool = [g for g in continuation_games if g.game_id not in pilot_ids]

    r = match_games_to_cells(pool, demands, sampling_seed)
    if not r.complete:
        return {
            "verdict": "ASSIGNMENT_SHORTFALL",
            "demands": {"|".join(k): v for k, v in demands.items()},
            "raw_capacity": {"|".join(k): sum(
                1 for g in pool if (k[1], k[2]) in eligible_cells(g))
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_corpus_geometry.py -v -p no:cacheprovider`
Expected: PASS — 33 passed (the demands test is parametrized over 6 cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/corpus_geometry.py tests/test_corpus_geometry.py
git commit -m "feat(atlas-s2): final assignment, ply selection and the failure artifact"
```

---

### Task 6: The CLI, and the operator stop

**Files:**
- Create: `scripts/GPU/alphazero/build_atlas_corpus.py`
- Test: `tests/test_build_atlas_corpus_cli.py`

**Interfaces:**
- Consumes: all of `corpus_geometry`.
- Produces: subcommands `emit-protocol`, `emit-gen-command`, `pilot-gate`, `size`, `assign`. **`emit-gen-command` prints the command and exits; it never runs it.**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_atlas_corpus_cli.py
import json
import subprocess
import sys


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.build_atlas_corpus", *args],
        capture_output=True, text=True, check=False,
    )


def test_emit_protocol_records_the_frozen_parameters():
    r = _run("emit-protocol", "--n-target", "240", "--base-seed", "20400000",
             "--sampling-seed", "20260804")
    assert r.returncode == 0, r.stderr
    p = json.loads(r.stdout)
    assert p["n_target"] == 240
    assert p["max_seed_range_games"] == 480
    assert p["seed_range"] == [20400000, 20400480]
    assert p["one_position_per_game"] is True
    assert p["no_top_up"] is True


def test_emit_gen_command_prints_but_does_not_run():
    r = _run("emit-gen-command", "--n-target", "240", "--base-seed", "20400000",
             "--sampling-seed", "20260804")
    assert r.returncode == 0, r.stderr
    assert "OPERATOR STOP" in r.stdout
    assert "NOT AUTHORIZED" in r.stdout
    # It must not have produced anything.
    assert "generated" not in r.stdout.lower()


def test_cli_refuses_a_disallowed_n():
    r = _run("emit-protocol", "--n-target", "250", "--base-seed", "20400000",
             "--sampling-seed", "1")
    assert r.returncode != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_build_atlas_corpus_cli.py -v -p no:cacheprovider`
Expected: FAIL — `No module named scripts.GPU.alphazero.build_atlas_corpus`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/build_atlas_corpus.py
"""Atlas corpus builder CLI -- design section 3.

THIS TOOL GENERATES NOTHING. `emit-gen-command` prints the generation command
and stops; running it is a separate operator authorization, exactly as Phase 0's
preflight was. No GPU work happens here.
"""
from __future__ import annotations

import argparse
import json
import sys

from .corpus_geometry import ALLOWED_N, MAX_SEED_RANGE_GAMES, PILOT_GAMES


def _protocol(args) -> dict:
    return {
        "n_target": args.n_target,
        "base_seed": args.base_seed,
        "seed_range": [args.base_seed, args.base_seed + MAX_SEED_RANGE_GAMES],
        "max_seed_range_games": MAX_SEED_RANGE_GAMES,
        "pilot_games": PILOT_GAMES,
        "sampling_seed": args.sampling_seed,
        "one_position_per_game": True,
        "no_top_up": True,
        "selection_inputs": ["game_id", "phase", "side", "sampling_seed"],
        "note": "Selection reads NO search result. Generation is a separate "
                "authorization and is not performed by this tool.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Atlas corpus builder (generates nothing)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("emit-protocol", "emit-gen-command"):
        s = sub.add_parser(name)
        s.add_argument("--n-target", type=int, required=True)
        s.add_argument("--base-seed", type=int, required=True)
        s.add_argument("--sampling-seed", type=int, required=True)
    args = ap.parse_args()

    if args.n_target not in ALLOWED_N:
        print(f"error: --n-target must be one of {ALLOWED_N}", file=sys.stderr)
        return 2

    if args.cmd == "emit-protocol":
        print(json.dumps(_protocol(args), indent=2, sort_keys=True))
        return 0

    print("=" * 72)
    print("OPERATOR STOP -- reservoir generation is NOT AUTHORIZED by this tool.")
    print("Review the protocol, obtain authorization, then run the command below.")
    print("=" * 72)
    print(json.dumps(_protocol(args), indent=2, sort_keys=True))
    print("\n# generation command (NOT run here) -- flags verified against"
          " generate_games.py --help:")
    print(f"#   .venv/bin/python -m scripts.GPU.alphazero.generate_games \\")
    print(f"#     --n-games <G_total from `size`> --seed {args.base_seed} \\")
    print(f"#     --simulations 400 --max-moves 280 --weights <checkpoint> \\")
    print(f"#     --output <path>")
    print("#   NOTE: shipped self-play generation keeps Dirichlet noise ON, so"
          " --no-noise is NOT passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> Flags verified against `generate_games.py --help`: `--n-games`, `--seed`, `--simulations`, `--max-moves`, `--weights`, `--output`, `--no-noise`. **`--no-noise` is deliberately NOT passed** — §3 requires *unchanged shipped game generation*, which keeps root Dirichlet noise on. That differs from the atlas ladder's `add_noise=False`, and the two must not be conflated: the corpus is generated as shipped self-play, then probed without noise.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_build_atlas_corpus_cli.py -v -p no:cacheprovider`
Expected: PASS — 3 passed.

- [ ] **Step 5: Full suite, then commit**

```bash
.venv/bin/python -m pytest -p no:cacheprovider -q > /tmp/s2.out 2>&1; echo "REAL_EXIT=$?" >> /tmp/s2.out; tail -3 /tmp/s2.out
git add scripts/GPU/alphazero/build_atlas_corpus.py tests/test_build_atlas_corpus_cli.py
git commit -m "feat(atlas-s2): corpus builder CLI that emits the protocol and stops"
```

Read `REAL_EXIT` from the file. **Never trust a `| tail` exit code** — it reports the pipe's status, which masked a collection error twice in Stage 1.

---

## Stage 2 completion criteria

- [ ] Phases, side derivation and eligibility use only `(n_moves, start_player)`.
- [ ] Stable digest ordering; no Python `hash()` anywhere.
- [ ] Matching is deterministic, respects one-position-per-game, and catches shared capacity.
- [ ] Pilot geometry gate passes and no-gos correctly, with a min-cut witness.
- [ ] 254-subset sweep verified as `2⁸−2`; `G_total ≤ 480` whenever the gate passed.
- [ ] Final demands integral at every allowed `N`, summing to `N − 24`.
- [ ] Shortfall emits the failure artifact and **no partial corpus**.
- [ ] CLI prints the generation command and stops.
- [ ] Full suite green, exit code read from the process.

## Out of scope

No reservoir generation, no self-play, no GPU work, no replay, no ladder, no read-outs. Stage 3 is planned only after these interfaces exist and qualify.

## Known gap handed to Stage 3

Stage 2 qualifies the geometry on **synthetic** `GameMeta`. It never consumes a real `GameRecord`, so the mapping from a generated game to `GameMeta(game_id, seed, n_moves, start_player)` is designed but unexercised. Stage 3 must drive at least one **real** record through it — the v18 lesson that four contract defects lived exactly where a consumer met a hand-written surrogate of its producer.
