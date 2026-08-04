# Atlas Stage 3 — Warm-Prefix Replay and Additive Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and qualify the warm-prefix replay producer, the four-leg additive ladder, and the batch-safe 320 boundary consumer — closing every warm-root qualification deferred by Stages 1 and 2, without generating a reservoir or running a measurement.

**Architecture:** One module `warm_prefix_replay.py`. A replay driver forces a game's recorded moves through shipped 400-simulation searches on **one** `MCTS` carrying **one** `random.Random(replay_seed)`, stops before the sampled target's search, and hands the resulting warm root to an additive ladder. A `BatchSafeBoundaryObserver` consumes Stage 1's `on_flush_complete` to capture the boundary; Stage 1's `SelectionTracer` rides along, cleared at every `advance_root`. A thin CLI emits an artifact and a runtime projection, and runs nothing.

**Tech Stack:** Python 3, stdlib only. Tests: `.venv/bin/python -m pytest -p no:cacheprovider`, `FakeEvaluator`, `active_size=6`.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-03-convergence-atlas-design.md` §2b, §4, §8, §9. **§3–§12 are EXECUTION-FROZEN.**
- **No reservoir generation, no checkpoint loading, no MLX execution, no measurement run.** Every test uses `FakeEvaluator`.
- **No `mcts.py` change.** Stage 1's scoped exception already delivered every hook Stage 3 consumes.
- **Frozen seeding (§2b):** `replay_seed = base_seed + game_idx`, **verified against the sidecar**. **One** `random.Random(replay_seed)` per row, continued through every prefix search **and all four ladder legs**. **Never reseeded** per ply or per rung. Moves are forced, so no temperature or move-selection draws are consumed — only `_select_child`'s tie-break draws from the stream.
- **Frozen ladder (§4):** legs `+400 → +1,200 → +1,600 → +3,200`, nominal `B = 400 / 1,600 / 3,200 / 6,400`. Record `B` / `I` / `I + B` per leg.
- **Frozen boundary (§4):** the **first flush completion at or after 320 completed target-search backups**. Record `N_actual`, `overshoot = N_actual − 320`, `remaining = 400 − N_actual`, `flush_type`. `N_actual` **excludes** inherited `I`; assert `320 ≤ N_actual ≤ 400`.
- **Frozen forced-move semantics (§2b):** always advance the recorded legal move; absent child → `forced_child_visits = None`, fresh node, `inheritance_reset = True`; present child → exact integer visits including zero, `inheritance_reset = False`. **No "shallow" threshold.** Report `zero_effective_inheritance = (absent) or (visits == 0)` while preserving the absent-vs-present-zero pair. Record reset count, rate and last reset ply; **keep every row**.
- **Tracer cache lifetime:** `SelectionTracer.clear_node_cache()` at **every** `advance_root`. `MCTSNode` is unhashable so the cache keys `id(node)`, and a detached subtree frees ids for reuse.
- **Artifact boundaries reuse `_jsonable`** from `build_atlas_corpus`. Geometry and ladder types stay natural; only serialization normalizes. **Do not re-key** the geometry module.
- **Undefined statistics are `None`, never `0` and never `false`.**
- Commit after every task.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/GPU/alphazero/warm_prefix_replay.py` (create) | Replay driver, boundary observer, additive ladder, row artifact, runtime projection. |
| `scripts/GPU/alphazero/run_atlas_ladder.py` (create) | CLI: `emit-plan`, `project-runtime`, `replay-row`. Emits and stops. |
| `tests/test_warm_prefix_replay.py` (create) | Prefix, boundary, ladder, tracer, seed continuity. |
| `tests/test_atlas_ladder_integration.py` (create) | Real Stage 2 sidecar → replay consumer; `_jsonable` artifact; runtime projection. |

---

### Task 1: The replay prefix — forced moves, one RNG, inheritance reset

**Files:**
- Create: `scripts/GPU/alphazero/warm_prefix_replay.py`
- Test: `tests/test_warm_prefix_replay.py`

**Interfaces:**
- Consumes: `MCTS`, `MCTSNode` from `mcts`; `GameMeta` from `corpus_geometry`.
- Produces: `@dataclass PrefixStep(ply, forced_move, forced_child_visits, inheritance_reset, zero_effective_inheritance)`; `@dataclass PrefixResult(root, inherited_I, steps, reset_count, reset_rate, last_reset_ply)`; `replay_prefix(mcts, meta, move_history, target_ply, active_size) -> PrefixResult`.

> **The RNG belongs to the `MCTS` instance and is never touched here.** `replay_prefix` receives an already-constructed `MCTS` carrying `random.Random(replay_seed)`, so the same stream flows through the prefix and, later, every ladder leg. Constructing an `MCTS` inside this function would break §2b continuity.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_warm_prefix_replay.py
import random

import pytest

from scripts.GPU.alphazero.corpus_geometry import GameMeta
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.warm_prefix_replay import replay_prefix

from tests.eval_fakes import FakeEvaluator

BASE = 20400000
SIZE = 6


def shipped_cfg(n_simulations: int = 400) -> MCTSConfig:
    return MCTSConfig(n_simulations=n_simulations, eval_batch_size=14,
                      stall_flush_sims=48, pending_virtual_visits=8)


def _mcts(seed, n_simulations=400, **kw):
    return MCTS(FakeEvaluator(value=0.0), shipped_cfg(n_simulations),
                random.Random(seed), **kw)


def _legal_history(n_plies, size=SIZE):
    """A legal move sequence on an empty board: distinct cells, row-major."""
    return [(i // size, i % size) for i in range(n_plies)]


def _meta(game_idx=0, n_moves=12):
    return GameMeta(game_id=game_idx, seed=BASE + game_idx, n_moves=n_moves,
                    start_player="red")


def test_prefix_produces_a_NONZERO_inherited_root():
    """The gap Stage 1 could not close: every Stage 1 search started fresh, so
    I was always 0 and the subtraction was trivially correct."""
    m = _mcts(BASE)
    r = replay_prefix(m, _meta(), _legal_history(4), target_ply=3,
                      active_size=SIZE)
    assert r.inherited_I > 0
    assert r.root.visit_count == r.inherited_I


def test_prefix_stops_immediately_before_the_target_search():
    m = _mcts(BASE)
    r = replay_prefix(m, _meta(), _legal_history(4), target_ply=3,
                      active_size=SIZE)
    assert [s.ply for s in r.steps] == [0, 1, 2]      # searches at plies 0..2
    assert len(r.steps) == 3


def test_target_ply_zero_yields_a_cold_root():
    m = _mcts(BASE)
    r = replay_prefix(m, _meta(), _legal_history(4), target_ply=0,
                      active_size=SIZE)
    assert r.steps == [] and r.inherited_I == 0


def test_forced_move_present_records_exact_visits_not_a_threshold():
    m = _mcts(BASE)
    r = replay_prefix(m, _meta(), _legal_history(4), target_ply=3,
                      active_size=SIZE)
    for s in r.steps:
        if not s.inheritance_reset:
            assert isinstance(s.forced_child_visits, int)
            assert s.forced_child_visits >= 0
            # zero_effective_inheritance unions absent-or-zero WITHOUT losing
            # the underlying pair.
            assert s.zero_effective_inheritance == (s.forced_child_visits == 0)


def test_reset_statistics_are_recorded_and_no_row_is_dropped():
    m = _mcts(BASE)
    r = replay_prefix(m, _meta(), _legal_history(4), target_ply=3,
                      active_size=SIZE)
    assert r.reset_count == sum(1 for s in r.steps if s.inheritance_reset)
    assert r.reset_rate == pytest.approx(r.reset_count / len(r.steps))
    assert len(r.steps) == 3          # every step retained
    if r.reset_count == 0:
        assert r.last_reset_ply is None      # None, never -1 and never 0


def test_prefix_rejects_a_target_ply_outside_the_history():
    m = _mcts(BASE)
    with pytest.raises(ValueError):
        replay_prefix(m, _meta(), _legal_history(4), target_ply=9,
                      active_size=SIZE)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_warm_prefix_replay.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named '...warm_prefix_replay'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/warm_prefix_replay.py
"""Atlas warm-prefix replay and additive ladder -- design sections 2b and 4.

Phase 0 returned WARM_START_REQUIRED, so the atlas probes a root carrying
trajectory-compounded inheritance rather than a fresh one. This module replays a
corpus game's recorded moves through shipped searches, stops before the sampled
target, and runs the frozen four-leg additive ladder on the resulting warm root.

Immediate-parent replay is NOT sufficient and is not implemented: inheritance
compounds, so one parent search cannot reproduce a tree carried across a full
trajectory.

CPU-SAFE at import: no MLX, no scipy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .corpus_geometry import GameMeta
from .mcts import MCTSNode, encode_move, visit_leader_move

BOUNDARY_THRESHOLD = 320          # design section 4
LEG_INCREMENTS = (400, 1200, 1600, 3200)
NOMINAL_B = (400, 1600, 3200, 6400)


@dataclass
class PrefixStep:
    ply: int
    forced_move: int
    forced_child_visits: Optional[int]     # None == child absent, never 0
    inheritance_reset: bool
    zero_effective_inheritance: bool
    state_agrees: bool = False             # canonical agreement after advance


@dataclass
class PrefixResult:
    root: MCTSNode
    inherited_I: int
    steps: List[PrefixStep] = field(default_factory=list)
    reset_count: int = 0
    reset_rate: Optional[float] = None
    last_reset_ply: Optional[int] = None
    cache_clears: int = 0                  # one per advance_root, counted


def replay_prefix(mcts, meta: GameMeta, move_history: Sequence[Tuple[int, int]],
                  target_ply: int, active_size: int = 24) -> PrefixResult:
    """Force `move_history[:target_ply]` through shipped searches.

    `mcts` MUST already carry random.Random(replay_seed); this function never
    constructs one, because section 2b requires a single stream across the
    prefix and every ladder leg.
    """
    from .game.twixt_state import TwixtState

    if target_ply < 0 or target_ply > len(move_history):
        raise ValueError(
            f"target_ply {target_ply} outside history of {len(move_history)} moves")

    if meta.n_moves != len(move_history):
        raise ValueError(
            f"metadata says n_moves={meta.n_moves} but history has "
            f"{len(move_history)} moves; the sidecar and replay disagree")

    root = MCTSNode(state=TwixtState(active_size=active_size,
                                     to_move=meta.start_player))
    steps: List[PrefixStep] = []
    cache_clears = 0
    for ply in range(target_ply):
        mcts.search_from_root(root, add_noise=False, ply=ply)
        move = tuple(move_history[ply])

        # Section 2b step 2: legality, then state agreement after the advance.
        if move not in root.state.legal_moves():
            raise ValueError(
                f"ply {ply}: recorded move {move} is not legal in the replayed "
                f"state; the replay has diverged from the source game")
        expected_state = root.state.apply_move(move)

        child = root.children.get(encode_move(move[0], move[1]))
        visits = None if child is None else child.visit_count
        reset = child is None

        root = mcts.advance_root(root, move)
        # Canonical agreement over (to_move, pegs, bridges) -- an inherited
        # child's state must equal the independently applied move. A silent
        # divergence here would make every downstream measurement describe a
        # different game.
        if root.state != expected_state or hash(root.state) != hash(expected_state):
            raise ValueError(
                f"ply {ply}: state disagreement after advance_root; the "
                f"inherited child does not match the applied recorded move")

        steps.append(PrefixStep(
            ply=ply, forced_move=encode_move(move[0], move[1]),
            forced_child_visits=visits, inheritance_reset=reset,
            # Unions absent-or-zero WITHOUT collapsing the pair: the fields
            # above keep None and 0 distinct.
            zero_effective_inheritance=(visits is None or visits == 0),
            state_agrees=True,
        ))

        # Cache lifetime (design section 8): a detached subtree frees id()
        # values for reuse, so a longer-lived rank cache would silently return
        # another node's ranks. Cleared at EVERY advance, and counted so a test
        # can prove every boundary cleared -- final emptiness alone would pass
        # if only the last advance did.
        tracer = getattr(mcts, "_selection_observer", None)
        if tracer is not None and hasattr(tracer, "clear_node_cache"):
            tracer.clear_node_cache()
            cache_clears += 1

    resets = [s.ply for s in steps if s.inheritance_reset]
    return PrefixResult(
        root=root, inherited_I=root.visit_count, steps=steps,
        reset_count=len(resets),
        reset_rate=(len(resets) / len(steps)) if steps else None,
        last_reset_ply=(resets[-1] if resets else None),
        cache_clears=cache_clears,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_warm_prefix_replay.py -v -p no:cacheprovider`
Expected: PASS — 6 passed. `test_prefix_produces_a_NONZERO_inherited_root` is the
Stage 1 gap closed: a 1-ply prefix at `active_size=6` yields `I ≈ 21`.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/warm_prefix_replay.py tests/test_warm_prefix_replay.py
git commit -m "feat(atlas-s3): warm-prefix replay with forced moves and inheritance reset"
```

---

### Task 2: The batch-safe boundary observer

**Files:**
- Modify: `scripts/GPU/alphazero/warm_prefix_replay.py`
- Test: `tests/test_warm_prefix_replay.py`

**Interfaces:**
- Consumes: Stage 1's `MCTSFlushObserver` protocol (`on_flush_complete(flush_type, root)`).
- Produces: `@dataclass BoundaryRecord(N_actual, overshoot, remaining, flush_type)`; `class BatchSafeBoundaryObserver(inherited_I, threshold=320, leg_B=400)` with `.record -> Optional[BoundaryRecord]`.

> **`N_actual` excludes `I`.** Stage 1 deliberately passes no backup counter: `root.visit_count` already counts completed backups, so the consumer computes `N_actual = root.visit_count − I` from the `I` it captured before the leg. This is the subtraction Stage 1 could never exercise, because every Stage 1 search started at `I = 0`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_warm_prefix_replay.py
from scripts.GPU.alphazero.warm_prefix_replay import (
    BOUNDARY_THRESHOLD, BatchSafeBoundaryObserver,
)


class _FakeRoot:
    def __init__(self, visit_count):
        self.visit_count = visit_count


def test_boundary_fires_at_the_first_flush_at_or_after_320():
    obs = BatchSafeBoundaryObserver(inherited_I=21)
    for total in (21 + 300, 21 + 318, 21 + 326, 21 + 340):
        obs.on_flush_complete("full", _FakeRoot(total))
    rec = obs.record
    assert rec is not None
    assert rec.N_actual == 326                 # first at-or-after 320, not 340
    assert rec.overshoot == 6
    assert rec.remaining == 400 - 326
    assert rec.flush_type == "full"


def test_boundary_subtracts_inherited_I():
    """The Stage 1 gap: with I = 0 this subtraction is trivially correct and
    proves nothing."""
    obs = BatchSafeBoundaryObserver(inherited_I=137)
    obs.on_flush_complete("full", _FakeRoot(137 + 322))
    assert obs.record.N_actual == 322


def test_boundary_ignores_later_flushes_once_captured():
    obs = BatchSafeBoundaryObserver(inherited_I=0)
    obs.on_flush_complete("full", _FakeRoot(321))
    obs.on_flush_complete("tail", _FakeRoot(400))
    assert obs.record.N_actual == 321 and obs.record.flush_type == "full"


def test_boundary_is_none_until_the_threshold_is_reached():
    obs = BatchSafeBoundaryObserver(inherited_I=0)
    obs.on_flush_complete("full", _FakeRoot(300))
    assert obs.record is None          # None, never a zero-filled record


def test_tail_only_search_yields_zero_remaining():
    """The degenerate case the deployability rule exists to catch."""
    obs = BatchSafeBoundaryObserver(inherited_I=0)
    obs.on_flush_complete("tail", _FakeRoot(400))
    assert obs.record.N_actual == 400 and obs.record.remaining == 0


def test_boundary_asserts_the_frozen_range():
    obs = BatchSafeBoundaryObserver(inherited_I=0)
    with pytest.raises(AssertionError):
        obs.on_flush_complete("full", _FakeRoot(401))


def test_boundary_in_a_REAL_400_sim_leg_on_a_warm_root():
    """End-to-end: a real batched search on a genuinely inherited root."""
    m = _mcts(BASE)
    pre = replay_prefix(m, _meta(), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    assert pre.inherited_I > 0
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    # SAME MCTS, so the single frozen RNG stream continues from the prefix --
    # a fresh MCTS would reset it, which section 2b forbids. The observer is
    # attached only now, so it cannot capture a boundary from a PREFIX search.
    m._flush_observer = obs
    m.search_from_root(pre.root, add_noise=False, ply=2)
    rec = obs.record
    assert rec is not None
    assert BOUNDARY_THRESHOLD <= rec.N_actual <= 400
    assert rec.remaining == 400 - rec.N_actual
    assert rec.flush_type in {"full", "stall", "tail"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_warm_prefix_replay.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'BatchSafeBoundaryObserver'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/GPU/alphazero/warm_prefix_replay.py
@dataclass
class BoundaryRecord:
    N_actual: int
    overshoot: int
    remaining: int
    flush_type: str


class BatchSafeBoundaryObserver:
    """Captures the FIRST flush completion at or after `threshold` completed
    TARGET-search backups (design section 4).

    A raw backup count will not do: at backup 320 the search is mid-flush, with
    expansions from later batch members already visible and up to
    eval_batch_size simulations queued as unredirectable waiters. Right after a
    flush's clears, the in-flight set is provably empty -- the only quiescent
    point in the loop.
    """

    def __init__(self, inherited_I: int, threshold: int = BOUNDARY_THRESHOLD,
                 leg_B: int = 400, tracer=None) -> None:
        if inherited_I < 0:
            raise ValueError("inherited_I must be non-negative")
        self._I = inherited_I
        self._threshold = threshold
        self._leg_B = leg_B
        self._tracer = tracer
        self.record: Optional[BoundaryRecord] = None
        # Section 8's FIRST frozen snapshot, taken at exactly the quiescent
        # boundary moment. Taking it later would describe a different tree.
        self.tracer_snapshot_at_boundary: Optional[Dict[str, Any]] = None

    def on_flush_complete(self, flush_type: str, root: Any) -> None:
        if self.record is not None:
            return                                # first one wins
        n_actual = root.visit_count - self._I      # excludes inherited visits
        if n_actual < self._threshold:
            return
        assert self._threshold <= n_actual <= self._leg_B, (
            f"N_actual {n_actual} outside the frozen "
            f"[{self._threshold}, {self._leg_B}] range")
        self.record = BoundaryRecord(
            N_actual=n_actual,
            overshoot=n_actual - self._threshold,
            remaining=self._leg_B - n_actual,
            flush_type=flush_type,
        )
        if self._tracer is not None:
            self.tracer_snapshot_at_boundary = self._tracer.snapshot()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_warm_prefix_replay.py -v -p no:cacheprovider`
Expected: PASS — 13 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/warm_prefix_replay.py tests/test_warm_prefix_replay.py
git commit -m "feat(atlas-s3): batch-safe boundary observer with I-excluding N_actual"
```

---

### Task 3: The four-leg additive ladder

**Files:**
- Modify: `scripts/GPU/alphazero/warm_prefix_replay.py`
- Test: `tests/test_warm_prefix_replay.py`

**Interfaces:**
- Consumes: `PrefixResult`, `BatchSafeBoundaryObserver`.
- Produces: `@dataclass LegResult(nominal_B, inherited_I, effective, root_value, selected_move, selected_move_prior_rank, top_share, top_two_margin, effective_children, n_visited_children, visit_counts)`; `_root_summary(root, visit_counts, selected_move) -> dict`; `run_additive_ladder(mcts, root, inherited_I, ply, boundary_observer=None, target_tracer=None, increments=LEG_INCREMENTS) -> tuple[list[LegResult], dict]` — the second element carries §8's `at_boundary` and `at_400` snapshots.

Legs are **additive on the same tree**: `+400 → +1,200 → +1,600 → +3,200`, giving nominal `B = 400 / 1,600 / 3,200 / 6,400` for **6,400 new simulations total** — cheaper than the fresh-root design's 11,600, because legs share accumulated work.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_warm_prefix_replay.py
from scripts.GPU.alphazero.mcts import encode_move, visit_leader_move
from scripts.GPU.alphazero.warm_prefix_replay import (
    LEG_INCREMENTS, NOMINAL_B, run_additive_ladder,
)


def test_leg_increments_sum_to_the_frozen_nominal_budgets():
    assert LEG_INCREMENTS == (400, 1200, 1600, 3200)
    assert NOMINAL_B == (400, 1600, 3200, 6400)
    running = 0
    for inc, b in zip(LEG_INCREMENTS, NOMINAL_B):
        running += inc
        assert running == b
    assert sum(LEG_INCREMENTS) == 6400


def test_ladder_records_B_I_and_effective_separately():
    m = _mcts(BASE, n_simulations=1)          # per-leg budget is set by the ladder
    pre = replay_prefix(m, _meta(), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    legs, _snaps = run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                                       boundary_observer=obs,
                                       increments=(4, 4, 4, 4))   # tiny, for speed
    assert [l.nominal_B for l in legs] == [4, 8, 12, 16]
    assert all(l.inherited_I == pre.inherited_I for l in legs)
    assert all(l.effective == l.inherited_I + l.nominal_B for l in legs)


def test_ladder_is_additive_on_one_tree_not_four_searches():
    """Root visits accumulate; each leg continues the previous tree."""
    m = _mcts(BASE, n_simulations=1)
    pre = replay_prefix(m, _meta(), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                        boundary_observer=obs, increments=(4, 4, 4, 4))
    assert pre.root.visit_count == pre.inherited_I + 16


def test_every_rung_preserves_its_own_evidence():
    """After leg 4 the tree is at 6,400 and the earlier rungs are GONE. Each
    LegResult must already carry what sections 5 and 7 need."""
    m = _mcts(BASE, n_simulations=1)
    pre = replay_prefix(m, _meta(), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    legs, _ = run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                                  boundary_observer=obs,
                                  increments=(8, 8, 8, 8))
    for leg in legs:
        assert leg.visit_counts and all(v > 0 for v in leg.visit_counts.values())
        assert leg.n_visited_children == len(leg.visit_counts)
        assert leg.top_share is not None
        assert leg.effective_children is not None
        assert leg.selected_move_prior_rank is not None
    # The distributions genuinely differ between rungs -- otherwise preserving
    # them would be pointless.
    assert legs[0].visit_counts != legs[-1].visit_counts
    assert sum(legs[0].visit_counts.values()) < sum(legs[-1].visit_counts.values())


def test_tracer_snapshots_are_taken_at_DISTINCT_times_in_a_real_400_leg():
    """Section 8's two frozen snapshots, on the grounded real case.

    A tiny 8-simulation leg would make the tail the ONLY qualifying flush, so
    both snapshots would capture the same instant and `a <= b` would pass while
    proving nothing. At 400 simulations with batch 14 there are ~28 full flushes,
    so the boundary lands near 320 with real budget left after it.
    """
    m = _mcts(BASE, n_simulations=1)
    pre = replay_prefix(m, _meta(), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    target_tracer = SelectionTracer()          # FRESH, attached after the prefix
    m._selection_observer = target_tracer
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I,
                                    tracer=target_tracer)
    _legs, snaps = run_additive_ladder(
        m, pre.root, pre.inherited_I, ply=2, boundary_observer=obs,
        target_tracer=target_tracer,
        increments=(400, 4, 4, 4))       # leg 1 REAL; later legs tiny for speed

    assert obs.record is not None
    assert obs.record.remaining > 0, (
        "the boundary landed on the tail flush, so both snapshots describe the "
        "same instant and this test would be vacuous")
    assert snaps["at_boundary"] is not None and snaps["at_400"] is not None
    a = snaps["at_boundary"]["by_shape"]["c4a05"]["overall"]["eligible_events"]
    b = snaps["at_400"]["by_shape"]["c4a05"]["overall"]["eligible_events"]
    assert 0 < a < b, "snapshots must be taken at DISTINCT times"


def test_ladder_rejects_a_wrong_length_increment_tuple():
    m = _mcts(BASE, n_simulations=1)
    pre = replay_prefix(m, _meta(), _legal_history(4), target_ply=1,
                        active_size=SIZE)
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    with pytest.raises(ValueError):
        run_additive_ladder(m, pre.root, pre.inherited_I, ply=1,
                            boundary_observer=obs, increments=(4, 4))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_warm_prefix_replay.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'run_additive_ladder'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/GPU/alphazero/warm_prefix_replay.py
@dataclass
class LegResult:
    """A rung's evidence, captured BEFORE the tree advances past it.

    The ladder is additive on ONE tree, so by the end of leg 4 the 400/1,600/3,200
    states no longer exist anywhere. Everything sections 5 and 7 need must be
    frozen here or it is unrecoverable: section 5 wants V_B and the 6,400 top-two
    margin; section 7 wants effective children, top share, and the selected
    move's prior RANK for the lower-prior-flip gate.
    """
    nominal_B: int
    inherited_I: int
    effective: int
    root_value: float
    selected_move: Optional[int]
    selected_move_prior_rank: Optional[int]
    top_share: Optional[float]
    top_two_margin: Optional[float]
    effective_children: Optional[float]
    n_visited_children: int
    visit_counts: Dict[int, int]           # compact: NONZERO entries only


def _root_summary(root: MCTSNode, visit_counts: Dict[Any, int],
                  selected_move: Optional[int]) -> Dict[str, Any]:
    """Derive section 5 / section 7 metrics from a root, at this rung.

    Undefined statistics are None, never 0.0 -- an empty or single-child
    distribution has no top-two margin, and that is a different fact from a
    margin of zero.
    """
    import math

    nonzero = {encode_move(r, c): v for (r, c), v in visit_counts.items() if v > 0}
    total = sum(nonzero.values())
    ordered = sorted(nonzero.values(), reverse=True)

    top_share = (ordered[0] / total) if (ordered and total) else None
    top_two_margin = (((ordered[0] - ordered[1]) / total)
                      if (len(ordered) >= 2 and total) else None)
    if total:
        # exp(entropy) -- the section 7 "effective children" metric.
        ent = -sum((v / total) * math.log(v / total) for v in nonzero.values())
        eff_children = math.exp(ent)
    else:
        eff_children = None

    # Prior rank of the selected move: adjusted prior DESCENDING, move-ID
    # ASCENDING -- the same frozen order the selection tracer uses.
    rank = None
    if selected_move is not None and root.priors:
        order = sorted(root.priors.items(), key=lambda kv: (-kv[1], kv[0]))
        for i, (mv, _p) in enumerate(order, start=1):
            if mv == selected_move:
                rank = i
                break

    return {
        "visit_counts": nonzero,
        "n_visited_children": len(nonzero),
        "top_share": top_share,
        "top_two_margin": top_two_margin,
        "effective_children": eff_children,
        "selected_move_prior_rank": rank,
    }


def run_additive_ladder(mcts, root: MCTSNode, inherited_I: int, ply: int,
                        boundary_observer=None, target_tracer=None,
                        increments: Sequence[int] = LEG_INCREMENTS
                        ) -> Tuple[List[LegResult], Dict[str, Any]]:
    """Four ADDITIVE legs on ONE tree. `mcts` keeps its RNG throughout.

    Returns (legs, snapshots). `snapshots` carries section 8's two frozen
    target-search tracer snapshots: "at_boundary" (taken by the boundary
    observer at N_actual) and "at_400" (taken here immediately after leg 1).

    `target_tracer` MUST be a fresh tracer attached AFTER prefix replay. A tracer
    that ran through the prefix would have its counters contaminated by unrelated
    searches, and section 8's statistics are about the TARGET search alone.

    The boundary observer is attached for leg 1 only: the 320-completion prefix
    lives inside the first 400-simulation leg (design section 4).
    """
    if len(increments) != 4:
        raise ValueError(f"the frozen ladder has four legs, got {len(increments)}")

    if target_tracer is not None:
        # Section 8's statistics are about the TARGET search alone. A tracer that
        # ran through prefix replay carries counters from unrelated searches, and
        # nothing downstream could tell the difference -- so refuse it here.
        if getattr(mcts, "_selection_observer", None) is not target_tracer:
            raise ValueError(
                "target_tracer must be the MCTS's current selection observer; "
                "attach it AFTER prefix replay")
        pre = target_tracer.snapshot()
        if any(pre["by_shape"][s]["overall"]["eligible_events"]
               for s in pre["by_shape"]):
            raise ValueError(
                "target_tracer is not empty: it already accumulated events, so "
                "its snapshots would be contaminated by prefix replay")

    legs: List[LegResult] = []
    snapshots: Dict[str, Any] = {"at_boundary": None, "at_400": None}
    running_B = 0
    original_n = mcts.config.n_simulations
    original_flush_obs = getattr(mcts, "_flush_observer", None)
    try:
        for leg_idx, inc in enumerate(increments):
            mcts.config.n_simulations = inc
            # Leg 1 only -- the 320 prefix is inside the first leg.
            mcts._flush_observer = boundary_observer if leg_idx == 0 else None
            visit_counts, root_value, root = mcts.search_from_root(
                root, add_noise=False, ply=ply)
            running_B += inc
            # CANONICAL leader: max visits, ties by lowest encoded move id.
            # max(visit_counts, key=...) would break ties by dict insertion
            # order, so stable-reference labels and lower-prior-flip metrics
            # could change between runs. Section 9 names this trap explicitly.
            sel = visit_leader_move(root)
            summary = _root_summary(root, visit_counts, sel)
            legs.append(LegResult(
                nominal_B=running_B, inherited_I=inherited_I,
                effective=inherited_I + running_B,
                root_value=float(root_value), selected_move=sel,
                selected_move_prior_rank=summary["selected_move_prior_rank"],
                top_share=summary["top_share"],
                top_two_margin=summary["top_two_margin"],
                effective_children=summary["effective_children"],
                n_visited_children=summary["n_visited_children"],
                visit_counts=summary["visit_counts"],
            ))
            if leg_idx == 0 and target_tracer is not None:
                # Section 8: the SECOND frozen snapshot, at nominal B = 400.
                snapshots["at_400"] = target_tracer.snapshot()
    finally:
        mcts.config.n_simulations = original_n
        mcts._flush_observer = original_flush_obs
    if boundary_observer is not None:
        snapshots["at_boundary"] = boundary_observer.tracer_snapshot_at_boundary
    return legs, snapshots
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_warm_prefix_replay.py -v -p no:cacheprovider`
Expected: PASS — 20 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/warm_prefix_replay.py tests/test_warm_prefix_replay.py
git commit -m "feat(atlas-s3): four-leg additive ladder recording B, I and I+B"
```

---

### Task 4: Tracer integration — cache clearing and forced-simulation tracing

**Files:**
- Test: `tests/test_warm_prefix_replay.py`

**Interfaces:**
- Consumes: Stage 1's `SelectionTracer`, `replay_prefix`, `run_additive_ladder`.
- Produces: no new API — this task **proves** two Stage 1 deferrals against a real warm replay.

Stage 1 exercised `clear_node_cache()` only by direct unit call, so the `id()`-reuse hazard it exists to prevent was never reproduced. It also observed `within_forced_simulation` only on synchronous forced simulations, never inside a warm replay.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_warm_prefix_replay.py
from scripts.GPU.alphazero.selection_tracer import SelectionTracer


def test_tracer_cache_is_cleared_at_every_real_advance_root():
    """Stage 1 could only call clear_node_cache() directly. Here the clears
    happen at genuine advance_root boundaries, where detached subtrees free
    id() values for reuse -- the hazard the cache lifetime exists to prevent."""
    tracer = SelectionTracer()
    m = _mcts(BASE, selection_observer=tracer)
    replay_prefix(m, _meta(), _legal_history(5), target_ply=4, active_size=SIZE)
    # After the final advance_root the cache holds nothing stale.
    assert tracer._cache == {}


def test_target_tracer_is_attached_AFTER_the_prefix():
    """The prefix runs with NO target tracer, so its events never enter section
    8's counters."""
    m = _mcts(BASE, n_simulations=1)
    pre = replay_prefix(m, _meta(), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    tracer = SelectionTracer()                    # FRESH, after the prefix
    m._selection_observer = tracer
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I, threshold=4,
                                    leg_B=8, tracer=tracer)
    run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                        boundary_observer=obs, target_tracer=tracer,
                        increments=(8, 8, 8, 8))
    snap = tracer.snapshot()
    for shape in ("c4a05", "c13a03"):
        cell = snap["by_shape"][shape]["overall"]
        assert cell["eligible_events"] > 0
        assert cell["outside_rate"] is not None      # denominator is non-empty


def test_ladder_refuses_a_contaminated_tracer():
    """A tracer that ran through the prefix must be rejected, not silently used
    -- nothing downstream could tell the difference."""
    tracer = SelectionTracer()
    m = _mcts(BASE, n_simulations=1, selection_observer=tracer)
    pre = replay_prefix(m, _meta(), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    with pytest.raises(ValueError, match="not empty"):
        run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                            boundary_observer=obs, target_tracer=tracer,
                            increments=(4, 4, 4, 4))


def test_ladder_refuses_a_tracer_that_is_not_the_selection_observer():
    m = _mcts(BASE, n_simulations=1)
    pre = replay_prefix(m, _meta(), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I)
    with pytest.raises(ValueError, match="selection observer"):
        run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                            boundary_observer=obs,
                            target_tracer=SelectionTracer(),   # never attached
                            increments=(4, 4, 4, 4))


def test_cache_is_cleared_ONCE_PER_ADVANCE_not_just_at_the_end():
    """Final emptiness alone would pass if only the LAST advance cleared it."""
    tracer = SelectionTracer()
    m = _mcts(BASE, selection_observer=tracer)
    r = replay_prefix(m, _meta(), _legal_history(5), target_ply=4,
                      active_size=SIZE)
    assert r.cache_clears == len(r.steps) == 4


def test_prefix_asserts_canonical_state_agreement_at_every_ply():
    m = _mcts(BASE)
    r = replay_prefix(m, _meta(), _legal_history(4), target_ply=3,
                      active_size=SIZE)
    assert all(s.state_agrees for s in r.steps)


def test_prefix_rejects_an_illegal_recorded_move():
    m = _mcts(BASE)
    history = _legal_history(4)
    history[2] = history[0]              # repeat an occupied cell
    with pytest.raises(ValueError, match="not legal"):
        replay_prefix(m, _meta(), history, target_ply=3, active_size=SIZE)


def test_prefix_rejects_metadata_that_disagrees_with_the_history():
    m = _mcts(BASE)
    with pytest.raises(ValueError, match="disagree"):
        replay_prefix(m, _meta(n_moves=99), _legal_history(4), target_ply=2,
                      active_size=SIZE)


def test_within_forced_simulation_is_observed_during_a_warm_replay():
    """Stage 1 saw this covariate only on synchronous forced simulations."""
    tracer = SelectionTracer()
    m = _mcts(BASE, selection_observer=tracer)
    replay_prefix(m, _meta(), _legal_history(3), target_ply=2, active_size=SIZE)
    snap = tracer.snapshot()
    # A warm replay uses the BATCHED path, where no simulation is forced, so the
    # covariate must be present and zero -- not absent.
    assert "within_forced_events" in snap
    assert snap["within_forced_events"] == 0
```

- [ ] **Step 2: Run and confirm**

Run: `.venv/bin/python -m pytest tests/test_warm_prefix_replay.py -v -p no:cacheprovider`
Expected: PASS — 29 passed.

If `test_tracer_cache_is_cleared_at_every_real_advance_root` fails, `replay_prefix` is not clearing at the `advance_root` boundary. Fix the driver; **do not** relax the assertion — the whole point is that a stale `id()` silently returns another node's ranks.

- [ ] **Step 3: Commit**

```bash
git add tests/test_warm_prefix_replay.py
git commit -m "test(atlas-s3): prove tracer cache clearing and forced covariate in warm replay"
```

---

### Task 5: Exact replay-seed continuity across prefix and all four legs

**Files:**
- Test: `tests/test_warm_prefix_replay.py`

**Interfaces:**
- Consumes: everything above.
- Produces: no new API — this proves §2b's continuity requirement.

> Reseeding per leg would leave the legs additive on the tree but draw tie-breaks from a **restarted** stream, silently breaking nesting so 6,400 stops being a true superset of 3,200. This task makes that failure detectable.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_warm_prefix_replay.py
def _row(seed, increments=(4, 4, 4, 4), reseed_between_legs=False):
    """Drive the legs directly rather than through run_additive_ladder.

    This task tests the RNG STREAM, not the ladder wrapper (Task 3 covers that),
    and driving searches directly avoids relying on zero-simulation legs to
    capture intermediate states.
    """
    m = MCTS(FakeEvaluator(value=0.0), shipped_cfg(1), random.Random(seed))
    pre = replay_prefix(m, _meta(), _legal_history(4), target_ply=2,
                        active_size=SIZE)
    states = [m.rng.getstate()]
    legs = []
    root = pre.root
    for i, inc in enumerate(increments):
        if reseed_between_legs and i > 0:
            m.rng = random.Random(seed)          # the FORBIDDEN behaviour
        m.config.n_simulations = inc
        vc, rv, root = m.search_from_root(root, add_noise=False, ply=2)
        # A local summary, NOT a LegResult: this task tests the RNG stream, and
        # coupling it to the ladder's dataclass shape would break it whenever
        # that shape changes -- as it just did.
        legs.append({
            "nominal_B": sum(increments[:i + 1]),
            "root_value": float(rv),
            "leader": visit_leader_move(root),          # canonical, not max()
            "visits": {encode_move(r, c): v for (r, c), v in vc.items() if v > 0},
        })
        states.append(m.rng.getstate())
    return legs, states


def test_the_rng_stream_advances_and_is_never_reseeded():
    _legs, states = _row(BASE)
    # Every checkpoint differs: the single stream is genuinely consumed.
    assert len({repr(s) for s in states}) == len(states)


def test_a_row_reproduces_exactly_under_the_same_replay_seed():
    a_legs, a_states = _row(BASE)
    b_legs, b_states = _row(BASE)
    assert [l["root_value"] for l in a_legs] == [l["root_value"] for l in b_legs]
    assert [l["leader"] for l in a_legs] == [l["leader"] for l in b_legs]
    assert [l["visits"] for l in a_legs] == [l["visits"] for l in b_legs]
    assert repr(a_states) == repr(b_states)


def test_reseeding_between_legs_CHANGES_the_result():
    """Non-vacuity: if continuity did not matter, this control would be
    indistinguishable and the continuity test would prove nothing."""
    ok_legs, _ = _row(BASE)
    bad_legs, _ = _row(BASE, reseed_between_legs=True)
    assert ([l["visits"] for l in ok_legs] != [l["visits"] for l in bad_legs]
            or [l["leader"] for l in ok_legs] != [l["leader"] for l in bad_legs]
            or [l["root_value"] for l in ok_legs] != [l["root_value"] for l in bad_legs])


def test_replay_seed_must_match_the_sidecar():
    """Section 2b: replay_seed = base_seed + game_idx, VERIFIED against the
    sidecar, not merely assumed."""
    from scripts.GPU.alphazero.warm_prefix_replay import replay_seed_for
    meta = _meta(game_idx=7)
    assert replay_seed_for(meta, base_seed=BASE) == BASE + 7
    bad = GameMeta(game_id=7, seed=BASE + 999, n_moves=12, start_player="red")
    with pytest.raises(ValueError, match="seed"):
        replay_seed_for(bad, base_seed=BASE)
```

- [ ] **Step 2: Add `replay_seed_for`**

```python
# append to scripts/GPU/alphazero/warm_prefix_replay.py
def replay_seed_for(meta: GameMeta, base_seed: int) -> int:
    """Section 2b: replay_seed = base_seed + game_idx, VERIFIED against the
    sidecar's recorded seed rather than assumed."""
    want = base_seed + meta.game_id
    if meta.seed != want:
        raise ValueError(
            f"game {meta.game_id}: sidecar seed {meta.seed} != base_seed + "
            f"game_idx ({want}); the frozen replay seed identity is violated")
    return want
```

- [ ] **Step 3: Run and commit**

Run: `.venv/bin/python -m pytest tests/test_warm_prefix_replay.py -v -p no:cacheprovider`
Expected: PASS — 33 passed.

```bash
git add scripts/GPU/alphazero/warm_prefix_replay.py tests/test_warm_prefix_replay.py
git commit -m "test(atlas-s3): prove replay-seed continuity across prefix and all legs"
```

---

### Task 6: Real Stage 2 sidecar → replay consumer integration

**Files:**
- Create: `tests/test_atlas_ladder_integration.py`

**Interfaces:**
- Consumes: Stage 2's `generate_block` / `load_block` / `game_meta_from_sidecar`; Stage 3's replay.
- Produces: no new API — this drives the **real** Stage 2 producer into the **real** Stage 3 consumer.

> The v18 lesson, and the reason this task exists as its own step: four contract defects lived exactly where a consumer met a hand-written surrogate of its producer. Stage 2 closed the seam for `GameMeta`; this closes it for `move_history` and the replay driver.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_ladder_integration.py
"""Real Stage 2 producer -> real Stage 3 consumer. No surrogates."""
import json
import random
from pathlib import Path

from scripts.GPU.alphazero.build_atlas_corpus import _jsonable
from scripts.GPU.alphazero.generate_atlas_reservoir import (
    game_meta_from_sidecar, generate_block,
)
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.selection_tracer import SelectionTracer
from scripts.GPU.alphazero.warm_prefix_replay import (
    BatchSafeBoundaryObserver, replay_prefix, replay_seed_for,
    run_additive_ladder,
)

from tests.eval_fakes import FakeEvaluator

BASE = 20400000
SIZE = 6
FAKE_PROV = {"git_head": "d" * 40, "worktree_clean": True,
             "checkpoint_path": "fake://evaluator", "checkpoint_sha1": "0" * 40}


def _real_block(tmp_path):
    return generate_block(
        evaluator=FakeEvaluator(value=0.0), base_seed=BASE, start_index=0,
        n_games=2, out_dir=Path(tmp_path), provenance=dict(FAKE_PROV),
        n_simulations=8, max_moves=10, active_size=SIZE,
    )


def test_a_real_sidecar_drives_the_real_replay(tmp_path):
    rows = _real_block(tmp_path / "blk")
    side = json.loads(
        (tmp_path / "blk" / "game_000000.json").read_text())
    meta = game_meta_from_sidecar(side)

    seed = replay_seed_for(meta, base_seed=BASE)      # verifies against sidecar
    assert seed == side["seed"]

    history = [tuple(m) for m in side["move_history"]]
    target_ply = min(2, len(history))
    m = MCTS(FakeEvaluator(value=0.0),
             MCTSConfig(n_simulations=8, eval_batch_size=14,
                        stall_flush_sims=48, pending_virtual_visits=8),
             random.Random(seed))
    pre = replay_prefix(m, meta, history, target_ply=target_ply,
                        active_size=SIZE)
    assert pre.inherited_I >= 0
    assert len(pre.steps) == target_ply


def test_the_replayed_prefix_follows_the_recorded_moves_exactly(tmp_path):
    rows = _real_block(tmp_path / "blk2")
    side = json.loads((tmp_path / "blk2" / "game_000001.json").read_text())
    meta = game_meta_from_sidecar(side)
    history = [tuple(m) for m in side["move_history"]]
    m = MCTS(FakeEvaluator(value=0.0),
             MCTSConfig(n_simulations=8, eval_batch_size=14,
                        stall_flush_sims=48, pending_virtual_visits=8),
             random.Random(replay_seed_for(meta, BASE)))
    pre = replay_prefix(m, meta, history, target_ply=3, active_size=SIZE)
    from scripts.GPU.alphazero.mcts import encode_move
    assert [s.forced_move for s in pre.steps] == [
        encode_move(*history[i]) for i in range(3)]


def test_the_row_artifact_survives_jsonable(tmp_path):
    """Ladder and boundary types must serialize through the SAME converter
    Stage 2 uses. Re-keying the geometry module is forbidden."""
    rows = _real_block(tmp_path / "blk3")
    side = json.loads((tmp_path / "blk3" / "game_000000.json").read_text())
    meta = game_meta_from_sidecar(side)
    history = [tuple(m) for m in side["move_history"]]
    m = MCTS(FakeEvaluator(value=0.0),
             MCTSConfig(n_simulations=1, eval_batch_size=14,
                        stall_flush_sims=48, pending_virtual_visits=8),
             random.Random(replay_seed_for(meta, BASE)))
    pre = replay_prefix(m, meta, history, target_ply=2, active_size=SIZE)
    tracer = SelectionTracer()                 # FRESH, after the prefix
    m._selection_observer = tracer
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I, threshold=4,
                                    leg_B=8, tracer=tracer)
    legs, snaps = run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                                      boundary_observer=obs,
                                      target_tracer=tracer,
                                      increments=(8, 8, 8, 8))
    artifact = {
        "game_idx": meta.game_id, "replay_seed": meta.seed,
        "inherited_I": pre.inherited_I,
        "reset_count": pre.reset_count, "reset_rate": pre.reset_rate,
        "last_reset_ply": pre.last_reset_ply,
        "legs": [vars(l) for l in legs],
        "boundary": (vars(obs.record) if obs.record else None),
        "tracer_snapshots": snaps,
    }
    text = json.dumps(_jsonable(artifact), sort_keys=True)
    back = json.loads(text)
    assert back["inherited_I"] == pre.inherited_I
    assert len(back["legs"]) == 4
    # Non-null snapshots must survive _jsonable -- serializing None would test
    # nothing about the tracer payload's serializability.
    assert back["tracer_snapshots"]["at_boundary"] is not None
    assert back["tracer_snapshots"]["at_400"] is not None
    assert back["legs"][0]["visit_counts"]        # per-rung evidence round-trips
```

- [ ] **Step 2: Run and commit**

Run: `.venv/bin/python -m pytest tests/test_atlas_ladder_integration.py -v -p no:cacheprovider`
Expected: PASS — 3 passed.

```bash
git add tests/test_atlas_ladder_integration.py
git commit -m "test(atlas-s3): real Stage 2 sidecar into the real replay consumer"
```

---

### Task 7: Runtime projection and the CLI

**Files:**
- Create: `scripts/GPU/alphazero/run_atlas_ladder.py`
- Modify: `tests/test_atlas_ladder_integration.py`

**Interfaces:**
- Consumes: `warm_prefix_replay`, `_jsonable`.
- Produces: `project_runtime(rows, mean_prefix_plies, tracer_overhead=0.010) -> dict`; CLI `emit-plan`, `project-runtime`. **No `replay-row` execution against a checkpoint** — that is a later authorization.

> **Use Stage 1's measured `+1.0%`**, not a guess. It is an **upper bound**: `FakeEvaluator` isolates tracer cost from NN cost, so a real evaluator's inference dominates and the true overhead is lower.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_atlas_ladder_integration.py
from scripts.GPU.alphazero.warm_prefix_replay import (
    LEG_INCREMENTS, project_runtime,
)


def test_runtime_projection_uses_stage1_measured_overhead():
    r = project_runtime(rows=240, mean_prefix_plies=69)
    # Ladder is 6,400 new sims per row; prefix is mean_plies * 400.
    assert r["ladder_sims_per_row"] == sum(LEG_INCREMENTS) == 6400
    assert r["prefix_sims_per_row"] == 69 * 400
    assert r["total_sims"] == 240 * (6400 + 69 * 400)
    assert r["tracer_overhead"] == 0.010
    assert r["tracer_overhead_is_upper_bound"] is True


def test_projection_states_the_prefix_dominates():
    r = project_runtime(rows=240, mean_prefix_plies=69)
    assert r["prefix_sims_per_row"] > r["ladder_sims_per_row"]
    assert r["dominant_term"] == "prefix_replay"


def test_projection_refuses_to_invent_a_ply_distribution():
    """mean_prefix_plies must be MEASURED from the corpus, never defaulted."""
    with pytest.raises(TypeError):
        project_runtime(rows=240)          # no default
```

- [ ] **Step 2: Write `project_runtime` and the CLI**

```python
# append to scripts/GPU/alphazero/warm_prefix_replay.py
def project_runtime(rows: int, mean_prefix_plies: float,
                    tracer_overhead: float = 0.010) -> Dict[str, Any]:
    """Simulation-count projection for the atlas run.

    `tracer_overhead` defaults to Stage 1's MEASURED +1.0% (400 sims, min-of-5,
    FakeEvaluator). It is an UPPER BOUND: FakeEvaluator isolates tracer cost from
    NN cost, so a real evaluator's inference dominates and the true figure is
    lower. `mean_prefix_plies` has no default -- it must be measured from the
    corpus's actual ply distribution, never assumed (design section 4).
    """
    ladder = sum(LEG_INCREMENTS)
    prefix = int(round(mean_prefix_plies * 400))
    return {
        "rows": rows,
        "ladder_sims_per_row": ladder,
        "prefix_sims_per_row": prefix,
        "sims_per_row": ladder + prefix,
        "total_sims": rows * (ladder + prefix),
        "dominant_term": "prefix_replay" if prefix > ladder else "ladder",
        "tracer_overhead": tracer_overhead,
        "tracer_overhead_is_upper_bound": True,
        "note": "Prefix cost scales with target ply and is dominated by late "
                "rows. Derive mean_prefix_plies from the corpus's observed "
                "per-phase ply supply, never from a smoke.",
    }
```

```python
# scripts/GPU/alphazero/run_atlas_ladder.py
"""Atlas ladder CLI -- design sections 2b and 4.

THIS TOOL RUNS NO MEASUREMENT. It emits the replay plan and a runtime
projection; executing the ladder against a real checkpoint is a separate
operator authorization.
"""
from __future__ import annotations

import argparse
import json
import sys

from .build_atlas_corpus import _jsonable
from .warm_prefix_replay import LEG_INCREMENTS, NOMINAL_B, project_runtime

_STOP = ("=" * 72 + "\nOPERATOR STOP -- the atlas measurement run is NOT "
         "AUTHORIZED by this tool.\n" + "=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="Atlas ladder (runs no measurement)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("emit-plan")
    s.add_argument("--corpus-artifact", required=True)

    s = sub.add_parser("project-runtime")
    s.add_argument("--rows", type=int, required=True)
    s.add_argument("--mean-prefix-plies", type=float, required=True)

    args = ap.parse_args()
    if args.cmd == "project-runtime":
        print(json.dumps(_jsonable(
            project_runtime(args.rows, args.mean_prefix_plies)),
            indent=2, sort_keys=True))
        return 0

    print(_STOP)
    print(json.dumps(_jsonable({
        "corpus_artifact": args.corpus_artifact,
        "leg_increments": list(LEG_INCREMENTS),
        "nominal_B": list(NOMINAL_B),
        "boundary": "first flush completion at or after 320 target-search backups",
        "add_noise": False,
        "note": "One random.Random(base_seed + game_idx) per row, continued "
                "across the prefix and all four legs. Never reseeded.",
    }), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Full suite, then commit**

```bash
.venv/bin/python -m pytest -p no:cacheprovider -q > /tmp/s3.out 2>&1; echo "REAL_EXIT=$?" >> /tmp/s3.out; tail -3 /tmp/s3.out
git add scripts/GPU/alphazero/warm_prefix_replay.py scripts/GPU/alphazero/run_atlas_ladder.py tests/
git commit -m "feat(atlas-s3): runtime projection from Stage 1's measured overhead, plus CLI"
```

Read `REAL_EXIT` from the file. **Never trust a `| tail` exit code.**

---

## Stage 3 completion criteria

- [ ] Prefix produces a **nonzero** inherited `I`, and `N_actual = root.visit_count − I` is correct with `I > 0`.
- [ ] Boundary fires at the **first flush at or after 320** target-search backups, with `320 ≤ N_actual ≤ 400` asserted, and `remaining == 0` reachable on a tail-only search.
- [ ] Ladder legs are additive on one tree, recording `B` / `I` / `I + B`.
- [ ] **Every rung preserves its own evidence before the tree advances past it** — nonzero visit counts, `n_visited_children`, `top_share`, `top_two_margin`, `effective_children`, and the selected move's prior **rank**. After leg 4 the earlier rungs no longer exist anywhere, and §5/§7 cannot be recomputed from the mutated tree.
- [ ] **Two frozen tracer snapshots** (§8): at `N_actual`, taken by the boundary observer at the quiescent moment, and at nominal `B = 400`, taken after leg 1. The tracer is **fresh and attached after prefix replay**, and `run_additive_ladder` **refuses** a tracer that is non-empty or is not the MCTS's selection observer.
- [ ] Snapshot timing proven on a **real 400-simulation leg**, asserting `remaining > 0` and `boundary_events < B400_events`. A tiny leg makes the tail the only qualifying flush, so both snapshots would describe the same instant and the check would be vacuous.
- [ ] Selected moves use the canonical `visit_leader_move` (ties by lowest encoded move id), **never** `max()` over dict order.
- [ ] Prefix asserts move **legality** and **canonical state agreement** (`to_move`, `pegs`, `bridges`) after every advance, and rejects metadata disagreeing with the history.
- [ ] Cache clears are **counted, one per advance** — final emptiness alone would pass if only the last advance cleared.
- [ ] Tracer cache cleared at **real** `advance_root` boundaries; `within_forced_simulation` observed during a warm replay.
- [ ] Replay-seed continuity proven across prefix and all four legs, **with a non-vacuity control** showing that reseeding changes the result.
- [ ] `replay_seed_for` verifies against the sidecar rather than assuming.
- [ ] A **real** Stage 2 sidecar drives the **real** replay consumer.
- [ ] Row artifact survives `_jsonable`; geometry is not re-keyed.
- [ ] Runtime projection uses Stage 1's measured `+1.0%`, labelled an upper bound, with no default `mean_prefix_plies`.
- [ ] Full suite green, exit code read from the process.

## Out of scope

No reservoir generation, no checkpoint loading, no MLX execution, no measurement run, no read-out logic. Stage 4 is planned only after these interfaces exist and qualify.
