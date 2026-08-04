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
| `scripts/GPU/alphazero/generate_atlas_reservoir.py` (create) | The producer. `game_seed = base_seed + game_idx`, `--start-index` continuation, per-game sidecars, provenance. |
| `scripts/GPU/alphazero/build_atlas_corpus.py` (create) | CLI: **all five** of `emit-protocol`, `emit-gen-command`, `pilot-gate`, `size`, `assign`. Emits commands; runs none. |
| `tests/test_corpus_geometry.py` (create) | Phases, eligibility, ordering, matching, min-cut, pilot gate, sizing — all on synthetic metadata. |
| `tests/test_generate_atlas_reservoir.py` (create) | Seed identity, continuation offset, single-index reproduction, sidecar contents. |
| `tests/test_build_atlas_corpus_cli.py` (create) | All five subcommands, the operator stop, exit codes, failure artifacts. |

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

### Task 6: The atlas reservoir producer

**Files:**
- Create: `scripts/GPU/alphazero/generate_atlas_reservoir.py`
- Test: `tests/test_generate_atlas_reservoir.py`

**Interfaces:**
- Consumes: `play_game` from `self_play`, `GameMeta` from `corpus_geometry`.
- Produces: CLI `--base-seed --start-index --n-games --checkpoint --out-dir`; per-game sidecars; a run-level provenance record; `game_meta_from_sidecar(d) -> GameMeta`.

> **Why the shipped generator cannot be used.** `generate_games.py` passes one master
> seed to `play_games`, which derives each game's RNG as
> `random.Random(rng.randint(0, 2**31))` (`self_play.py:1485-1490`). Game *i*'s seed
> therefore depends on **every preceding draw**, so:
>
> - there is **no start offset** — a continuation block at index 24 is unreachable
>   without replaying the first 24 draws;
> - a rerun at `base_seed` **regenerates the pilot** rather than the continuation;
> - `GameRecord` carries **neither index nor seed**, so §2b's
>   `replay_seed = reservoir_base_seed + game_idx` is unsatisfiable and there is no
>   sidecar for Stage 3 to verify against.
>
> This producer exists solely to restore the frozen per-game seed identity. It changes
> nothing about how a game is played.

**Shipped-equivalence requirement.** `play_games` derives `start_player` by consuming
one draw from the game RNG *before* playing (`"red" if game_rng.random() < 0.5 else
"black"`). Reproduce that **exactly** — same draw, same order — so each game is
shipped-identical given its RNG. Only the RNG's *provenance* changes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate_atlas_reservoir.py
import json
import random
from pathlib import Path

from scripts.GPU.alphazero.generate_atlas_reservoir import (
    game_meta_from_sidecar, generate_block, seed_for_index,
)
from scripts.GPU.alphazero.corpus_geometry import GameMeta

from tests.eval_fakes import FakeEvaluator


def test_seed_is_base_plus_index_exactly():
    assert seed_for_index(20400000, 0) == 20400000
    assert seed_for_index(20400000, 24) == 20400024
    assert seed_for_index(20400000, 479) == 20400479


def _block(tmp_path, start_index, n_games):
    return generate_block(
        evaluator=FakeEvaluator(value=0.0), base_seed=20400000,
        start_index=start_index, n_games=n_games, out_dir=Path(tmp_path),
        n_simulations=8, max_moves=6, active_size=6,
    )


def test_continuation_block_is_disjoint_and_offset(tmp_path):
    pilot = _block(tmp_path / "a", 0, 3)
    cont = _block(tmp_path / "b", 3, 3)
    assert [g["game_idx"] for g in pilot] == [0, 1, 2]
    assert [g["game_idx"] for g in cont] == [3, 4, 5]
    assert [g["seed"] for g in cont] == [20400003, 20400004, 20400005]
    assert not ({g["seed"] for g in pilot} & {g["seed"] for g in cont})


def test_a_single_index_reproduces_exactly(tmp_path):
    """The whole point: index 4 is the same game whether it is produced in a
    block starting at 0 or a continuation starting at 4."""
    from_zero = _block(tmp_path / "x", 0, 6)[4]
    from_four = _block(tmp_path / "y", 4, 1)[0]
    assert from_zero["seed"] == from_four["seed"] == 20400004
    assert from_zero["start_player"] == from_four["start_player"]
    assert from_zero["n_moves"] == from_four["n_moves"]
    assert from_zero["move_history"] == from_four["move_history"]


def test_sidecar_carries_everything_GameMeta_needs(tmp_path):
    rows = _block(tmp_path / "s", 0, 2)
    side = json.loads((Path(tmp_path / "s") / "game_000000.json").read_text())
    for k in ("game_idx", "seed", "start_player", "n_moves"):
        assert k in side
    meta = game_meta_from_sidecar(side)
    assert isinstance(meta, GameMeta)
    assert meta.game_id == 0 and meta.seed == 20400000


def test_provenance_record_is_written(tmp_path):
    _block(tmp_path / "p", 0, 2)
    prov = json.loads((Path(tmp_path / "p") / "provenance.json").read_text())
    assert prov["base_seed"] == 20400000
    assert prov["start_index"] == 0 and prov["n_games"] == 2
    assert prov["add_noise"] is True          # shipped generation keeps noise ON
    assert "git_head" in prov and "worktree_clean" in prov
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generate_atlas_reservoir.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named '...generate_atlas_reservoir'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/generate_atlas_reservoir.py
"""Atlas reservoir producer -- design section 3 source protocol and 2b seeding.

Exists because the shipped generator cannot satisfy the frozen per-game seed
identity: play_games derives each game RNG from a MASTER stream
(random.Random(rng.randint(...))), so game i's seed depends on every preceding
draw -- there is no start offset, a continuation block is unreachable, and the
emitted GameRecord carries neither index nor seed.

Here: game_seed = base_seed + game_idx, exactly. A block is therefore fully
determined by (base_seed, start_index, n_games), and any single index reproduces
independently of the block it was produced in.

How a game is PLAYED is unchanged: same play_game, shipped settings, Dirichlet
noise ON, and start_player derived by the same leading draw play_games uses.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .corpus_geometry import GameMeta, MAX_SEED_RANGE_GAMES
from .mcts import MCTSConfig
from .self_play import play_game


def seed_for_index(base_seed: int, game_idx: int) -> int:
    """The frozen identity. No master stream, no offset arithmetic elsewhere."""
    return base_seed + game_idx


def game_meta_from_sidecar(d: Dict[str, Any]) -> GameMeta:
    return GameMeta(game_id=d["game_idx"], seed=d["seed"],
                    n_moves=d["n_moves"], start_player=d["start_player"])


def _git(args: Sequence[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def generate_block(evaluator, base_seed: int, start_index: int, n_games: int,
                   out_dir: Path, n_simulations: int = 400,
                   max_moves: int = 280, active_size: int = 24) -> List[Dict[str, Any]]:
    """Generate games [start_index, start_index + n_games) and write sidecars."""
    if start_index < 0 or n_games <= 0:
        raise ValueError("start_index must be >= 0 and n_games > 0")
    if start_index + n_games > MAX_SEED_RANGE_GAMES:
        raise ValueError(
            f"block exceeds the frozen {MAX_SEED_RANGE_GAMES}-game seed range")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = MCTSConfig(n_simulations=n_simulations, eval_batch_size=14,
                     stall_flush_sims=48, pending_virtual_visits=8)
    rows: List[Dict[str, Any]] = []
    for game_idx in range(start_index, start_index + n_games):
        seed = seed_for_index(base_seed, game_idx)
        game_rng = random.Random(seed)
        # Same leading draw play_games consumes, in the same order, so a game is
        # shipped-identical GIVEN its RNG. Only the RNG's provenance changes.
        start_player = "red" if game_rng.random() < 0.5 else "black"
        rec = play_game(
            evaluator=evaluator, mcts_config=cfg, rng=game_rng,
            max_moves=max_moves, add_noise=True, active_size=active_size,
            start_player=start_player, game_id=game_idx,
        )
        row = {
            "game_idx": game_idx, "seed": seed, "start_player": start_player,
            "n_moves": rec.n_moves, "winner": rec.winner,
            "draw_reason": rec.draw_reason,
            "move_history": [list(m) for m in rec.move_history],
        }
        (out_dir / f"game_{game_idx:06d}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True))
        rows.append(row)

    (out_dir / "provenance.json").write_text(json.dumps({
        "base_seed": base_seed, "start_index": start_index, "n_games": n_games,
        "seed_range": [seed_for_index(base_seed, start_index),
                       seed_for_index(base_seed, start_index + n_games)],
        "n_simulations": n_simulations, "max_moves": max_moves,
        "active_size": active_size,
        "batching": [cfg.eval_batch_size, cfg.stall_flush_sims,
                     cfg.pending_virtual_visits],
        # Shipped generation keeps root Dirichlet noise ON. This is NOT the
        # atlas ladder's add_noise=False; the two must not be conflated.
        "add_noise": True,
        "git_head": _git(["rev-parse", "HEAD"]),
        "worktree_clean": _git(["status", "--porcelain"]) == "",
    }, indent=2, sort_keys=True))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Atlas reservoir producer")
    ap.add_argument("--base-seed", type=int, required=True)
    ap.add_argument("--start-index", type=int, required=True)
    ap.add_argument("--n-games", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--simulations", type=int, default=400)
    ap.add_argument("--max-moves", type=int, default=280)
    args = ap.parse_args()

    # One long-lived evaluator, shared across every game in the block. Rebuilding
    # a compiled evaluator per unit of work is the documented MLX trap.
    from .eval_runner import _default_evaluator_factory
    rows = generate_block(
        evaluator=_default_evaluator_factory(args.checkpoint),
        base_seed=args.base_seed, start_index=args.start_index,
        n_games=args.n_games, out_dir=Path(args.out_dir),
        n_simulations=args.simulations, max_moves=args.max_moves,
    )
    print(f"generated {len(rows)} games, indices "
          f"[{args.start_index}, {args.start_index + args.n_games})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generate_atlas_reservoir.py -v -p no:cacheprovider`
Expected: PASS — 5 passed.

`test_a_single_index_reproduces_exactly` is the load-bearing one: it proves the
continuation block is genuinely the continuation, which is exactly what the shipped
master-stream generator cannot provide.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/generate_atlas_reservoir.py tests/test_generate_atlas_reservoir.py
git commit -m "feat(atlas-s2): atlas reservoir producer with frozen per-game seeding"
```

---

### Task 7: The CLI — all five subcommands, and the operator stop

**Files:**
- Create: `scripts/GPU/alphazero/build_atlas_corpus.py`
- Test: `tests/test_build_atlas_corpus_cli.py`

**Interfaces:**
- Consumes: all of `corpus_geometry`; `game_meta_from_sidecar` from Task 6.
- Produces: `emit-protocol`, `emit-gen-command`, `pilot-gate`, `size`, `assign`. **All five, or the operator cannot invoke the qualified geometry.**

`pilot-gate`, `size` and `assign` read sidecar JSON produced by Task 6 and emit the
verdict or the failure artifact. `emit-gen-command` prints **two** commands — pilot
`[0, 24)` and continuation `[24, G_total)` — and runs neither.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_atlas_corpus_cli.py
import json
import subprocess
import sys
from pathlib import Path


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.build_atlas_corpus", *args],
        capture_output=True, text=True, check=False,
    )


def _sidecars(tmp_path, n, n_moves=200):
    d = Path(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"game_{i:06d}.json").write_text(json.dumps({
            "game_idx": i, "seed": 20400000 + i, "n_moves": n_moves,
            "start_player": "red" if i % 2 == 0 else "black",
        }))
    return str(d)


def test_emit_protocol_records_the_frozen_parameters():
    r = _run("emit-protocol", "--n-target", "240", "--base-seed", "20400000",
             "--sampling-seed", "20260804")
    assert r.returncode == 0, r.stderr
    p = json.loads(r.stdout)
    assert p["n_target"] == 240
    assert p["max_seed_range_games"] == 480
    assert p["seed_range"] == [20400000, 20400480]
    assert p["one_position_per_game"] is True and p["no_top_up"] is True


def test_emit_gen_command_prints_two_blocks_and_does_not_run():
    r = _run("emit-gen-command", "--n-target", "240", "--base-seed", "20400000",
             "--sampling-seed", "20260804", "--g-total", "280")
    assert r.returncode == 0, r.stderr
    assert "OPERATOR STOP" in r.stdout and "NOT AUTHORIZED" in r.stdout
    assert "--start-index 0 --n-games 24" in r.stdout      # pilot block
    assert "--start-index 24 --n-games 256" in r.stdout    # continuation
    assert "generate_atlas_reservoir" in r.stdout
    assert "generated" not in r.stdout.lower()


def test_pilot_gate_passes_on_long_games(tmp_path):
    d = _sidecars(tmp_path / "pg", 24)
    r = _run("pilot-gate", "--sidecar-dir", d, "--sampling-seed", "7")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["verdict"] == "PASS"


def test_pilot_gate_no_gos_on_short_games_with_exit_3(tmp_path):
    d = _sidecars(tmp_path / "pg2", 24, n_moves=60)
    r = _run("pilot-gate", "--sidecar-dir", d, "--sampling-seed", "7")
    out = json.loads(r.stdout)
    assert out["verdict"] == "PHASE_GEOMETRY_NO_GO"
    assert r.returncode == 3
    assert out["min_cut_cells"]


def test_size_reports_g_total_and_binding_subset(tmp_path):
    d = _sidecars(tmp_path / "sz", 24)
    r = _run("size", "--sidecar-dir", d, "--n-target", "240")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["verdict"] == "OK" and out["G_total"] <= 480
    assert out["binding_subset"]


def test_assign_emits_rows_or_a_failure_artifact(tmp_path):
    pilot = _sidecars(tmp_path / "p", 24)
    cont = _sidecars(tmp_path / "c", 400)
    r = _run("assign", "--pilot-dir", pilot, "--continuation-dir", cont,
             "--n-target", "200", "--sampling-seed", "7")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["verdict"] == "OK" and len(out["rows"]) == 176


def test_assign_shortfall_exits_4_with_no_partial_corpus(tmp_path):
    pilot = _sidecars(tmp_path / "p2", 24)
    cont = _sidecars(tmp_path / "c2", 400, n_moves=50)
    r = _run("assign", "--pilot-dir", pilot, "--continuation-dir", cont,
             "--n-target", "200", "--sampling-seed", "7")
    out = json.loads(r.stdout)
    assert out["verdict"] == "ASSIGNMENT_SHORTFALL"
    assert r.returncode == 4
    assert "rows" not in out and out["min_cut_cells"]


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

THIS TOOL GENERATES NOTHING. `emit-gen-command` prints the two generation
commands -- pilot and continuation -- and stops. Running them is a separate
operator authorization, exactly as Phase 0's preflight was.

Exit codes: 0 OK, 2 usage, 3 PHASE_GEOMETRY_NO_GO, 4 ASSIGNMENT_SHORTFALL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .corpus_geometry import (
    ALLOWED_N, MAX_SEED_RANGE_GAMES, PILOT_GAMES, GameMeta, assign_corpus,
    pilot_geometry_gate, size_continuation,
)
from .generate_atlas_reservoir import game_meta_from_sidecar


def _load(sidecar_dir: str) -> List[GameMeta]:
    metas = []
    for p in sorted(Path(sidecar_dir).glob("game_*.json")):
        metas.append(game_meta_from_sidecar(json.loads(p.read_text())))
    if not metas:
        raise SystemExit(f"no game_*.json sidecars in {sidecar_dir}")
    return metas


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
        if name == "emit-gen-command":
            s.add_argument("--g-total", type=int, required=True)
            s.add_argument("--checkpoint", default="<checkpoint>")
            s.add_argument("--out-dir", default="<out-dir>")

    s = sub.add_parser("pilot-gate")
    s.add_argument("--sidecar-dir", required=True)
    s.add_argument("--sampling-seed", type=int, required=True)

    s = sub.add_parser("size")
    s.add_argument("--sidecar-dir", required=True)
    s.add_argument("--n-target", type=int, required=True)

    s = sub.add_parser("assign")
    s.add_argument("--pilot-dir", required=True)
    s.add_argument("--continuation-dir", required=True)
    s.add_argument("--n-target", type=int, required=True)
    s.add_argument("--sampling-seed", type=int, required=True)

    args = ap.parse_args()
    if getattr(args, "n_target", None) is not None and args.n_target not in ALLOWED_N:
        print(f"error: --n-target must be one of {ALLOWED_N}", file=sys.stderr)
        return 2

    if args.cmd == "emit-protocol":
        print(json.dumps(_protocol(args), indent=2, sort_keys=True))
        return 0

    if args.cmd == "emit-gen-command":
        print("=" * 72)
        print("OPERATOR STOP -- reservoir generation is NOT AUTHORIZED by this tool.")
        print("Review the protocol, obtain authorization, then run BOTH blocks below.")
        print("=" * 72)
        print(json.dumps(_protocol(args), indent=2, sort_keys=True))
        base = (f".venv/bin/python -m scripts.GPU.alphazero.generate_atlas_reservoir"
                f" --base-seed {args.base_seed}")
        tail = (f" --checkpoint {args.checkpoint} --out-dir {args.out_dir}"
                f" --simulations 400 --max-moves 280")
        print("\n# 1. pilot block (NOT run here):")
        print(f"#   {base} --start-index 0 --n-games {PILOT_GAMES}{tail}")
        print("\n# 2. continuation block, AFTER the pilot gate passes (NOT run here):")
        print(f"#   {base} --start-index {PILOT_GAMES} "
              f"--n-games {args.g_total - PILOT_GAMES}{tail}")
        print("\n# Per-game seed is base_seed + game_idx, so the continuation block")
        print("# genuinely continues the pilot rather than regenerating it.")
        return 0

    if args.cmd == "pilot-gate":
        r = pilot_geometry_gate(_load(args.sidecar_dir), args.sampling_seed)
        print(json.dumps(r, indent=2, sort_keys=True, default=str))
        return 0 if r["verdict"] == "PASS" else 3

    if args.cmd == "size":
        r = size_continuation(_load(args.sidecar_dir), args.n_target)
        print(json.dumps(r, indent=2, sort_keys=True, default=str))
        return 0 if r["verdict"] == "OK" else 3

    # assign
    pilot = _load(args.pilot_dir)
    gate = pilot_geometry_gate(pilot, args.sampling_seed)
    if gate["verdict"] != "PASS":
        print(json.dumps(gate, indent=2, sort_keys=True, default=str))
        return 3
    r = assign_corpus(gate["assignment"], _load(args.continuation_dir),
                      args.n_target, args.sampling_seed)
    print(json.dumps(r, indent=2, sort_keys=True, default=str))
    return 0 if r["verdict"] == "OK" else 4


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_build_atlas_corpus_cli.py -v -p no:cacheprovider`
Expected: PASS — 8 passed.

- [ ] **Step 5: Full suite, then commit**

```bash
.venv/bin/python -m pytest -p no:cacheprovider -q > /tmp/s2.out 2>&1; echo "REAL_EXIT=$?" >> /tmp/s2.out; tail -3 /tmp/s2.out
git add scripts/GPU/alphazero/build_atlas_corpus.py tests/test_build_atlas_corpus_cli.py
git commit -m "feat(atlas-s2): corpus builder CLI, all five subcommands, operator stop"
```

Read `REAL_EXIT` from the file. **Never trust a `| tail` exit code** — it reports the
pipe's status, which masked a collection error twice in Stage 1.

## Stage 2 completion criteria

- [ ] Phases, side derivation and eligibility use only `(n_moves, start_player)`.
- [ ] Stable digest ordering; no Python `hash()` anywhere.
- [ ] Matching is deterministic, respects one-position-per-game, and catches shared capacity.
- [ ] Pilot geometry gate passes and no-gos correctly, with a min-cut witness.
- [ ] 254-subset sweep verified as `2⁸−2`; `G_total ≤ 480` whenever the gate passed.
- [ ] Final demands integral at every allowed `N`, summing to `N − 24`.
- [ ] Shortfall emits the failure artifact and **no partial corpus**.
- [ ] Producer: `game_seed = base_seed + game_idx`; a continuation block genuinely continues; any single index reproduces independently of its block.
- [ ] Sidecars carry `game_idx`, `seed`, `start_player`, `n_moves`; provenance records `add_noise=True`.
- [ ] CLI implements **all five** subcommands, prints **both** generation blocks, and stops. Exit codes 0/2/3/4.
- [ ] Full suite green, exit code read from the process.

## Out of scope

No reservoir generation, no self-play, no GPU work, no replay, no ladder, no read-outs. Stage 3 is planned only after these interfaces exist and qualify.

## Known gap handed to Stage 3

**Narrowed by Task 6.** The producer writes sidecars carrying exactly what `GameMeta`
needs, and `game_meta_from_sidecar` is exercised against a **real** `play_game` record
(with `FakeEvaluator`), so the record → `GameMeta` seam is no longer surrogate-only.

What remains for Stage 3: the producer is qualified at **tiny scale** — a few games,
8 simulations, `max_moves=6`, `active_size=6`, `FakeEvaluator`. It has never run at
shipped settings against a real checkpoint, so throughput, memory behaviour over a
480-game range, and the `q_S` a real ply distribution yields are all unmeasured. The
geometry is proven; the **supply** it will be handed is not.
