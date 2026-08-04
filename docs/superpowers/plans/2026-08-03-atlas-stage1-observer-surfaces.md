# Atlas Stage 1 — Observer Surfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the two frozen diagnostic observer surfaces to `mcts.py` and the selection tracer that consumes them, proving byte-identical-off behaviour and measuring tracer overhead.

**Architecture:** Two new Protocols with their own optional `MCTS.__init__` parameters, defaulting to `None`, so each hook is independently disableable and therefore independently qualifiable. `on_flush_complete` fires at all three flush-and-clear sites; `on_select_child` fires in both descent loops at the point where the override and PUCT branches converge on `(move_id, child)`, before lazy child creation. A separate `selection_tracer.py` accumulates counters online with a per-search prior-rank cache.

**Tech Stack:** Python 3, stdlib only. Tests are pytest, run as `.venv/bin/python -m pytest -p no:cacheprovider`.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-03-convergence-atlas-design.md`, §4, §8, §9. **§3–§12 are EXECUTION-FROZEN.** No parameter, threshold, predicate, producer or geometry rule may be changed.
- **Scoped `mcts.py` exception (§9), and nothing beyond it.** Diagnostic observer surfaces only: default `None`, read-only, no tree/search/RNG mutation. **No selection-rule, backup, config-default or budget change.**
- **`on_flush_complete` fires only AFTER the pending structures are cleared** — never mid-flush. That is the entire point of the batch-safe boundary.
- **Byte-identical off.** All observers `None` must leave search behaviour bit-for-bit unchanged. The whole pre-existing suite must pass unchanged.
- **The existing `MCTSObserver` and `FpuTraceObserver` must not break.** `FpuTraceObserver` (`diagnose_fpu_policy_mass.py:529`) implements only `on_root_simulation`; the new hooks must therefore be **separate Protocols with separate parameters**, never added as required methods to `MCTSObserver`.
- **Two emission sites for `on_select_child`, not one:** `search_from_root` (line 617) and `_run_single_simulation` (line 782). The latter is the forced-root-visit path (`_run_single_simulation(root, root_move_override=move_id)`, line 839). A hook only in the batched loop makes `root_override=True` unreachable.
- **Three flush sites:** full (658), stall (672), tail (681). Each clears `pending_nodes`, `pending_waiters`, `pending_node_ids` immediately after `_flush_pending_batch`.
- **Prior rank order:** adjusted prior **descending**, move-ID **ascending**.
- **`first_touch := existing_child is None`.** Present-with-zero-visits stays distinct and is never folded into first-touch.
- **Rank cache keys `id(node)` and clears at every `advance_root`.** `MCTSNode` is a plain `@dataclass` and therefore unhashable; subtree detach frees `id()` values for reuse, so a longer-lived cache silently returns another node's ranks.
- **No node mutation, no event objects, no event logs.** Counters only.
- **Zero denominators produce `None`, never `0` and never `false`.**
- **This stage authorizes no corpus generation and no measurement run.**
- Commit after every task.

---

## File Structure

| File | Responsibility |
|---|---|
| `tests/golden/atlas_stage1_prehook_search.json` (create, **Task 0**) | Golden captured from **unmodified** `mcts.py`. The byte-identity proof depends on this existing before any edit and cannot be reconstructed afterwards. |
| `scripts/GPU/alphazero/mcts.py` (modify) | Two Protocols, two constructor params, five call sites. Nothing else. |
| `scripts/GPU/alphazero/selection_tracer.py` (create) | The tracer: prior-rank cache, `K(n)` for the two frozen shapes, online counters. |
| `tests/atlas_stage1_fixtures.py` (create) | Shared fixtures. **Required**: the two test modules must NOT import from each other — that is a circular import which passes per-file and fails only under full-suite collection order. |
| `tests/test_atlas_observer_hooks.py` (create) | Hook firing, placement, `root_override` reachability, flush-site coverage. |
| `tests/test_selection_tracer.py` (create) | Rank order, cache lifetime, counter accumulation, `None` denominators. |
| `tests/test_atlas_observer_identity.py` (create) | Byte-identical-off, per-hook and all-on identity, timing smoke. |

---

### Task 0: Capture the pre-hook golden

**Files:**
- Create: `tests/golden/atlas_stage1_prehook_search.json`
- Test: `tests/test_atlas_observer_identity.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a golden JSON of a fixed batched search, captured from **unmodified** `mcts.py`.

> **This task MUST complete before any `mcts.py` edit.** The v16 tooling learned this: the identity proof depends on the ordering, and a golden captured after the edit proves nothing. Follow the existing precedent at `tests/golden/fpu_prebranch_search.json`.

- [ ] **Step 1: Write the capture-and-compare test**

```python
# tests/test_atlas_observer_identity.py
"""Stage 1 identity qualification. The golden in Step 2 is captured from
UNMODIFIED mcts.py; every later task re-proves it."""
import json
import random
from pathlib import Path

from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.game.twixt_state import TwixtState

from tests.eval_fakes import FakeEvaluator

GOLDEN = Path("tests/golden/atlas_stage1_prehook_search.json")


def shipped_config(n_simulations: int = 200) -> MCTSConfig:
    # Shipped batching is (14, 48, 8). stall_flush_sims defaults to 16, so 48
    # MUST be set explicitly or this is not the shipped batched path.
    return MCTSConfig(
        n_simulations=n_simulations,
        eval_batch_size=14,
        stall_flush_sims=48,
        pending_virtual_visits=8,
    )


def run_fixed_search(n_simulations: int = 200, **observer_kwargs):
    """One deterministic batched search. observer_kwargs are passed to MCTS.

    The 200 default is the GOLDEN's budget and must not change -- every identity
    comparison depends on it. The timing smoke passes 400 explicitly, because
    design section 8 requires a real 400-simulation measurement.
    """
    state = TwixtState(active_size=24, to_move="red")
    mcts = MCTS(
        FakeEvaluator(value=0.0),
        shipped_config(n_simulations),
        random.Random(20260803),
        **observer_kwargs,
    )
    visit_counts, root_value, root = mcts.search_from_root(
        _fresh_root(mcts, state), add_noise=False, ply=0
    )
    # NOTE: lists, not tuples -- json.dumps writes arrays, so a tuple-valued
    # fresh run would never compare equal to the JSON-decoded golden.
    return {
        "root_value": round(float(root_value), 12),
        "root_visit_count": root.visit_count,
        "visit_counts": sorted([f"{r}:{c}", v] for (r, c), v in visit_counts.items()),
        "flush_full": mcts._flush_full,
        "flush_stall": mcts._flush_stall,
        "flush_tail": mcts._flush_tail,
    }


def _fresh_root(mcts, state):
    from scripts.GPU.alphazero.mcts import MCTSNode
    return MCTSNode(state=state)


def test_golden_exists_and_reproduces():
    assert GOLDEN.exists(), (
        "Task 0 golden missing. It MUST be captured from UNMODIFIED mcts.py "
        "before any hook edit; capturing it later proves nothing."
    )
    assert run_fixed_search() == json.loads(GOLDEN.read_text())


def test_golden_is_not_vacuous():
    """A different seed must differ, or the comparison proves nothing."""
    state = TwixtState(active_size=24, to_move="red")
    mcts = MCTS(FakeEvaluator(value=0.0), shipped_config(), random.Random(999999))
    vc, _rv, _root = mcts.search_from_root(_fresh_root(mcts, state), add_noise=False, ply=0)
    other = sorted([f"{r}:{c}", v] for (r, c), v in vc.items())
    assert other != json.loads(GOLDEN.read_text())["visit_counts"]


def test_batched_path_was_exercised():
    """No full-batch flush means the golden says nothing about the batched path."""
    assert run_fixed_search()["flush_full"] > 0
```

- [ ] **Step 2: Run it to confirm it fails on the missing golden**

Run: `.venv/bin/python -m pytest tests/test_atlas_observer_identity.py -v -p no:cacheprovider`
Expected: FAIL — "Task 0 golden missing".

- [ ] **Step 3: Capture the golden from unmodified `mcts.py`**

```bash
git diff --stat scripts/GPU/alphazero/mcts.py   # MUST be empty before capturing
.venv/bin/python -c "
import json, pathlib
from tests.test_atlas_observer_identity import run_fixed_search
pathlib.Path('tests/golden/atlas_stage1_prehook_search.json').write_text(
    json.dumps(run_fixed_search(), indent=2, sort_keys=True))
print('golden captured')
"
```

If `git diff` on `mcts.py` is **not** empty, stop. The golden must come from unmodified source.

- [ ] **Step 4: Run to verify all three pass**

Run: `.venv/bin/python -m pytest tests/test_atlas_observer_identity.py -v -p no:cacheprovider`
Expected: PASS — 3 passed. If `test_batched_path_was_exercised` fails, raise `n_simulations` until a full-batch flush occurs; do not accept the golden without one.

- [ ] **Step 5: Commit**

```bash
git add tests/golden/atlas_stage1_prehook_search.json tests/test_atlas_observer_identity.py
git commit -m "test(atlas-s1): capture the pre-hook golden from unmodified mcts.py"
```

---

### Task 1: The two Protocols and their constructor parameters

**Files:**
- Modify: `scripts/GPU/alphazero/mcts.py` (Protocols beside `MCTSObserver` at line 269; constructor at 285)
- Test: `tests/test_atlas_observer_hooks.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MCTSFlushObserver` with `on_flush_complete(flush_type: str, root: MCTSNode) -> None`; `MCTSSelectionObserver` with `on_select_child(parent, selected_move, existing_child, depth, parent_completed_visits, root_override, within_forced_simulation) -> None`; `MCTS(..., flush_observer=None, selection_observer=None)`.

Separate Protocols and separate parameters, so each hook is independently `None`-able and therefore independently qualifiable. **Do not add methods to `MCTSObserver`** — `FpuTraceObserver` implements only `on_root_simulation` and would break.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_observer_hooks.py
import random

from scripts.GPU.alphazero.mcts import (
    MCTS, MCTSConfig, MCTSNode, MCTSFlushObserver, MCTSSelectionObserver,
)
from scripts.GPU.alphazero.game.twixt_state import TwixtState

from tests.eval_fakes import FakeEvaluator
from tests.test_atlas_observer_identity import shipped_config


class RecordingFlush:
    def __init__(self):
        self.calls = []

    def on_flush_complete(self, flush_type, root):
        self.calls.append((flush_type, root.visit_count))


class RecordingSelection:
    def __init__(self):
        self.calls = []

    def on_select_child(self, parent, selected_move, existing_child, depth,
                        parent_completed_visits, root_override,
                        within_forced_simulation):
        self.calls.append({
            "move": selected_move,
            "first_touch": existing_child is None,
            "depth": depth,
            "parent_visits": parent_completed_visits,
            "root_override": root_override,
            "forced_sim": within_forced_simulation,
        })


def test_mcts_accepts_both_new_observers_independently():
    ev, cfg = FakeEvaluator(value=0.0), shipped_config()
    MCTS(ev, cfg, random.Random(1), flush_observer=RecordingFlush())
    MCTS(ev, cfg, random.Random(1), selection_observer=RecordingSelection())
    MCTS(ev, cfg, random.Random(1),
         flush_observer=RecordingFlush(), selection_observer=RecordingSelection())


def test_existing_observer_protocol_is_untouched():
    """FpuTraceObserver implements ONLY on_root_simulation. Adding the new hooks
    to MCTSObserver would break it."""
    assert not hasattr(MCTSFlushObserver, "on_root_simulation")
    assert not hasattr(MCTSSelectionObserver, "on_root_simulation")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_observer_hooks.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'MCTSFlushObserver'`

- [ ] **Step 3: Write minimal implementation**

Beside `MCTSObserver` (line 269) in `mcts.py`:

```python
class MCTSFlushObserver(Protocol):
    """Read-only batch-flush completion hook (atlas design section 4).

    Fires ONLY after `_flush_pending_batch` and after `pending_nodes`,
    `pending_waiters` and `pending_node_ids` are cleared -- i.e. at a point
    where the in-flight set is provably empty. Never mid-flush.
    """
    def on_flush_complete(self, flush_type: str, root: "MCTSNode") -> None: ...

    # NOTE: no backup counter is passed, and `_backup` is NOT modified.
    # `root.visit_count` already counts completed backups -- every `_backup`
    # walks from the root -- so the consumer derives the section 4 quantity as
    # `N_actual = root.visit_count - I`, using the `I` it already records per
    # row. `_observer_completed_count` is unusable here: it is only INITIALIZED
    # when a root observer is attached (mcts.py:308), so reading it with only
    # the flush observer present raises AttributeError.


class MCTSSelectionObserver(Protocol):
    """Read-only per-selection hook (atlas design section 8).

    Fires after the move is resolved -- by PUCT or by a root override -- and
    BEFORE lazy child creation, so `existing_child is None` means first touch.
    """
    def on_select_child(self, parent: "MCTSNode", selected_move: int,
                        existing_child: Optional["MCTSNode"], depth: int,
                        parent_completed_visits: int, root_override: bool,
                        within_forced_simulation: bool) -> None: ...
```

In `MCTS.__init__` (line 285), after `observer`:

```python
        flush_observer: Optional["MCTSFlushObserver"] = None,
        selection_observer: Optional["MCTSSelectionObserver"] = None,
```

and in the body, beside `self._observer = observer`:

```python
        # Atlas diagnostic surfaces. None (default) mutates nothing and is part
        # of the byte-identical-off path (design section 9 scoped exception).
        self._flush_observer = flush_observer
        self._selection_observer = selection_observer
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_observer_hooks.py tests/test_atlas_observer_identity.py -v -p no:cacheprovider`
Expected: PASS — 5 passed. The Task 0 golden must still reproduce.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/mcts.py tests/test_atlas_observer_hooks.py
git commit -m "feat(atlas-s1): add flush and selection observer protocols, default None"
```

---

### Task 2: `on_flush_complete` at all three flush-and-clear sites

**Files:**
- Modify: `scripts/GPU/alphazero/mcts.py` (after the clears at 658–663, 672–677, 681–685)
- Test: `tests/test_atlas_observer_hooks.py`

**Interfaces:**
- Consumes: `MCTSFlushObserver` from Task 1.
- Produces: `on_flush_complete` firing with `flush_type` in `{"full", "stall", "tail"}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_atlas_observer_hooks.py


def _run(n_simulations=200, active_size=24, **kw):
    state = TwixtState(active_size=active_size, to_move="red")
    mcts = MCTS(FakeEvaluator(value=0.0), shipped_config(n_simulations),
                random.Random(20260803), **kw)
    root = MCTSNode(state=state)
    mcts.search_from_root(root, add_noise=False, ply=0)
    return mcts, root


def test_flush_hook_fires_and_types_match_counters():
    fo = RecordingFlush()
    mcts, _root = _run(flush_observer=fo)
    assert fo.calls, "no flush events emitted"
    seen = [t for t, _n in fo.calls]
    assert seen.count("full") == mcts._flush_full
    assert seen.count("stall") == mcts._flush_stall
    assert seen.count("tail") == mcts._flush_tail


def test_flush_events_are_ordered_and_tail_is_last():
    """What this CAN establish: events arrive in non-decreasing root-visit
    order and the tail flush is last.

    What it CANNOT: that `pending_nodes` / `pending_waiters` / `pending_node_ids`
    were empty at the callback. Those are locals inside `search_from_root` and
    are not observable from a test. After-the-clears placement is enforced as a
    CALL-SITE REQUIREMENT (Global Constraints, Task 2 Step 3) and by review --
    not by this test, and not by exposing internal queues."""
    fo = RecordingFlush()
    _run(flush_observer=fo)
    counts = [n for _t, n in fo.calls]
    assert counts == sorted(counts)
    assert fo.calls[-1][0] == "tail"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_observer_hooks.py -v -p no:cacheprovider`
Expected: FAIL — `AssertionError: no flush events emitted`

- [ ] **Step 3: Write minimal implementation**

At each of the three sites, immediately **after** `pending_node_ids.clear()` (and after `stall_count = 0` where present), add the matching block. Full flush (after line 663):

```python
                    if self._flush_observer is not None:
                        self._flush_observer.on_flush_complete("full", root)
```

Stall flush (after line 677): identical but `"stall"`. Tail flush (after line 685): identical but `"tail"`.

> **Do not pass `self._observer_completed_count`.** It is only *initialized* when a root observer is attached (`mcts.py:308`) and only incremented under the same condition (1108–1109), so reading it with just the flush observer present raises `AttributeError` — precisely the per-hook case Task 5 qualifies. **`_backup` is not modified.** The consumer derives `N_actual = root.visit_count − I` instead, which needs no new state because every `_backup` walks from the root.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_observer_hooks.py tests/test_atlas_observer_identity.py -v -p no:cacheprovider`
Expected: PASS — 7 passed, golden still reproducing.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/mcts.py tests/test_atlas_observer_hooks.py
git commit -m "feat(atlas-s1): fire on_flush_complete at all three flush-and-clear sites"
```

---

### Task 3: `on_select_child` in both descent loops, including forced root overrides

**Files:**
- Modify: `scripts/GPU/alphazero/mcts.py` (batched descent 617; synchronous descent 782)
- Test: `tests/test_atlas_observer_hooks.py`

**Interfaces:**
- Consumes: `MCTSSelectionObserver` from Task 1.
- Produces: `on_select_child` firing from both loops, with `root_override=True` reachable.

**Placement is exact.** In both loops the override branch and the PUCT branch converge on `(move_id, child)` before `if child is None`. Emit **there** — after resolution, before creation. A wrapper around `_select_child` would miss the override entirely, since it reads `node.children.get(move_id)` directly.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_atlas_observer_hooks.py


def test_selection_hook_fires_with_first_touch_and_depth():
    """Small board on purpose. At active_size=24 the FakeEvaluator's uniform
    priors over ~528 moves mean 200 sims never revisit a child, so EVERY event
    is a first touch and the present-child path goes unexercised."""
    so = RecordingSelection()
    _run(selection_observer=so, active_size=6)
    assert so.calls
    assert any(c["first_touch"] for c in so.calls)
    assert any(not c["first_touch"] for c in so.calls)
    assert min(c["depth"] for c in so.calls) == 0
    # Root selections see the root's own completed visit count.
    assert all(c["parent_visits"] >= 0 for c in so.calls)


def test_root_override_is_reachable():
    """_run_single_simulation(root_move_override=...) must emit an event with
    root_override=True. A hook only in the batched loop makes this unreachable."""
    so = RecordingSelection()
    state = TwixtState(active_size=24, to_move="red")
    mcts = MCTS(FakeEvaluator(value=0.0), shipped_config(8),
                random.Random(20260803), selection_observer=so)
    root = MCTSNode(state=state)
    mcts._expand(root)
    forced_move = sorted(root.priors)[0]
    mcts._run_single_simulation(root, root_move_override=forced_move)

    overrides = [c for c in so.calls if c["root_override"]]
    assert len(overrides) == 1, f"expected exactly one override event, got {len(overrides)}"
    assert overrides[0]["move"] == forced_move
    assert overrides[0]["depth"] == 0
    # Descendants of a forced simulation are normal PUCT and stay eligible.
    assert all(not c["root_override"] for c in so.calls if c["depth"] > 0)


def test_within_forced_simulation_is_a_covariate():
    so = RecordingSelection()
    state = TwixtState(active_size=24, to_move="red")
    mcts = MCTS(FakeEvaluator(value=0.0), shipped_config(8),
                random.Random(20260803), selection_observer=so)
    root = MCTSNode(state=state)
    mcts._expand(root)
    mcts._run_single_simulation(root, root_move_override=sorted(root.priors)[0])
    assert all(c["forced_sim"] for c in so.calls)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_observer_hooks.py -v -p no:cacheprovider`
Expected: FAIL — no selection events emitted.

- [ ] **Step 3: Write minimal implementation**

In `search_from_root`'s descent (after line 617's `_select_child`, before `if child is None`):

```python
                if self._selection_observer is not None:
                    self._selection_observer.on_select_child(
                        parent=node, selected_move=move_id, existing_child=child,
                        depth=len(search_path) - 1,
                        parent_completed_visits=node.visit_count,
                        root_override=False, within_forced_simulation=False,
                    )
```

In `_run_single_simulation`'s descent, after the `if override is not None: ... else: ...` block and before `if child is None`:

```python
            if self._selection_observer is not None:
                self._selection_observer.on_select_child(
                    parent=node, selected_move=move_id, existing_child=child,
                    depth=len(search_path) - 1,
                    parent_completed_visits=node.visit_count,
                    root_override=_was_override,
                    within_forced_simulation=within_forced,
                )
```

Set `_was_override = override is not None` **before** the branch consumes `override`, since the branch sets `override = None`. `within_forced` is `True` for the whole simulation when `root_move_override` was passed to `_run_single_simulation`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_observer_hooks.py tests/test_atlas_observer_identity.py -v -p no:cacheprovider`
Expected: PASS — 10 passed, golden still reproducing.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/mcts.py tests/test_atlas_observer_hooks.py
git commit -m "feat(atlas-s1): fire on_select_child in both descent loops incl. root override"
```

---

### Task 4: The selection tracer — rank cache, `K(n)`, online counters

**Files:**
- Create: `scripts/GPU/alphazero/selection_tracer.py`
- Test: `tests/test_selection_tracer.py`

**Interfaces:**
- Consumes: the `on_select_child` signature from Task 3.
- Produces: `WIDENING_SHAPES`; `k_of_n(n, c, alpha) -> int`; `n_admit(rank, c, alpha) -> int`; `class SelectionTracer` with `on_select_child(...)`, `clear_node_cache()`, `snapshot() -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_selection_tracer.py
import pytest

from scripts.GPU.alphazero.selection_tracer import (
    WIDENING_SHAPES, SelectionTracer, k_of_n, n_admit,
)


def test_frozen_shapes_are_pinned():
    assert WIDENING_SHAPES == (("c4a05", 4.0, 0.5), ("c13a03", 13.0, 0.3))


@pytest.mark.parametrize("n,expected", [(400, 80), (105, 41), (20, 18), (5, 9), (1, 4)])
def test_k_of_n_shape_a(n, expected):
    assert k_of_n(n, 4.0, 0.5, n_legal=10_000) == expected


@pytest.mark.parametrize("n,expected", [(400, 79), (105, 53), (20, 32), (5, 22), (1, 13)])
def test_k_of_n_shape_b(n, expected):
    assert k_of_n(n, 13.0, 0.3, n_legal=10_000) == expected


def test_k_is_clamped_by_n_legal_and_floored_at_one():
    assert k_of_n(400, 4.0, 0.5, n_legal=12) == 12
    assert k_of_n(0, 4.0, 0.5, n_legal=500) == 1


def test_n_admit_is_a_search_not_a_closed_form():
    """The closed form ceil((r/C)^(1/alpha)) is WRONG -- it discards the ceil
    inside K. At (C=4, alpha=0.5, r=9) it returns 6, but K(5)=9 >= 9."""
    assert n_admit(9, 4.0, 0.5, n_legal=500) == 5
    # Rank 1 is admitted at n=0 via the max(1, ...) floor.
    assert n_admit(1, 4.0, 0.5, n_legal=500) == 0


class _Node:
    def __init__(self, priors, visit_count=0):
        self.priors = priors
        self.visit_count = visit_count
        self.children = {}


def test_rank_order_is_prior_desc_then_move_id_asc():
    t = SelectionTracer()
    parent = _Node({7: 0.5, 3: 0.5, 9: 0.2})   # 3 and 7 tie on prior
    assert t._ranks_for(parent) == {3: 1, 7: 2, 9: 3}


def test_cache_is_cleared_on_demand():
    t = SelectionTracer()
    parent = _Node({1: 1.0})
    t._ranks_for(parent)
    assert t._cache
    t.clear_node_cache()
    assert not t._cache


def test_zero_denominator_yields_none_not_zero():
    snap = SelectionTracer().snapshot()
    for shape, _c, _a in WIDENING_SHAPES:
        cell = snap["by_shape"][shape]["overall"]
        assert cell["outside_rate"] is None
        assert cell["first_touch_outside_rate"] is None
        assert snap["by_shape"][shape]["meaningfully_affected"] is None


def test_forced_root_overrides_leave_the_primary_denominator():
    t = SelectionTracer()
    parent = _Node({1: 0.9, 2: 0.1}, visit_count=5)
    t.on_select_child(parent=parent, selected_move=2, existing_child=None, depth=0,
                      parent_completed_visits=5, root_override=True,
                      within_forced_simulation=True)
    snap = t.snapshot()
    for shape, _c, _a in WIDENING_SHAPES:
        assert snap["by_shape"][shape]["overall"]["eligible_events"] == 0
        assert snap["by_shape"][shape]["forced_root_bypass_events"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_selection_tracer.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named '...selection_tracer'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/selection_tracer.py
"""Atlas Read-out C producer -- design section 8, EXECUTION-FROZEN.

Accumulates counters ONLINE. No event objects, no event logs, no node mutation.
The per-node prior-rank cache keys `id(node)` because `MCTSNode` is a plain
`@dataclass` and therefore unhashable, and it MUST be cleared at every
`advance_root`: detaching a subtree frees `id()` values for reuse, so a
longer-lived cache would silently return another node's ranks.

CPU-SAFE: stdlib only.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# Frozen widening shapes (design section 8). Matched at the root, divergent below.
WIDENING_SHAPES: Tuple[Tuple[str, float, float], ...] = (
    ("c4a05", 4.0, 0.5),
    ("c13a03", 13.0, 0.3),
)

# Frozen: at least 10% of first-touch selections outside the admitted set.
MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE = 0.10

DEPTH_BUCKETS = ("0", "1", "2+")


def _bucket(depth: int) -> str:
    return "0" if depth == 0 else ("1" if depth == 1 else "2+")


def k_of_n(n: int, c: float, alpha: float, n_legal: int) -> int:
    """K(n) = min(n_legal, max(1, ceil(C * n^alpha)))."""
    return min(n_legal, max(1, math.ceil(c * (n ** alpha))))


def n_admit(rank: int, c: float, alpha: float, n_legal: int) -> int:
    """min { n >= 0 integer : K(n) >= rank }, computed as a SEARCH.

    The closed form ceil((rank/C)^(1/alpha)) is wrong: it inverts C*n^alpha >= r
    and discards the ceil inside K. Counterexample (C=4, alpha=0.5, r=9): the
    closed form gives 6, but K(5) = ceil(4*sqrt(5)) = 9 >= 9.
    """
    n = 0
    while k_of_n(n, c, alpha, n_legal) < rank:
        n += 1
    return n


def _empty_cell() -> Dict[str, int]:
    return {
        "eligible_events": 0,
        "outside_events": 0,
        "first_touch_events": 0,
        "first_touch_outside_events": 0,
        "excluded_prior_mass": 0.0,
    }


class SelectionTracer:
    """One tracer per row. Counters only."""

    def __init__(self) -> None:
        self._cache: Dict[int, Tuple[Dict[int, int], Dict[int, float]]] = {}
        self._cells: Dict[str, Dict[str, Dict[str, Any]]] = {
            shape: {b: _empty_cell() for b in ("overall",) + DEPTH_BUCKETS}
            for shape, _c, _a in WIDENING_SHAPES
        }
        self._forced_bypass: Dict[str, Dict[str, int]] = {
            shape: {"events": 0, "outside_events": 0}
            for shape, _c, _a in WIDENING_SHAPES
        }
        self._within_forced_events = 0

    # -- cache ---------------------------------------------------------
    def _ranks_for(self, parent: Any) -> Dict[int, int]:
        """Prior rank: adjusted prior DESCENDING, move-ID ASCENDING."""
        key = id(parent)
        hit = self._cache.get(key)
        if hit is None:
            ordered = sorted(parent.priors.items(), key=lambda kv: (-kv[1], kv[0]))
            ranks = {mv: i + 1 for i, (mv, _p) in enumerate(ordered)}
            priors = dict(parent.priors)
            self._cache[key] = (ranks, priors)
            hit = self._cache[key]
        return hit[0]

    def clear_node_cache(self) -> None:
        """MUST be called at every `advance_root` -- see the module docstring."""
        self._cache.clear()

    # -- hook ----------------------------------------------------------
    def on_select_child(self, parent, selected_move, existing_child, depth,
                        parent_completed_visits, root_override,
                        within_forced_simulation) -> None:
        ranks = self._ranks_for(parent)
        rank = ranks[selected_move]
        n_legal = len(parent.priors)
        first_touch = existing_child is None
        if within_forced_simulation:
            self._within_forced_events += 1

        for shape, c, alpha in WIDENING_SHAPES:
            k = k_of_n(parent_completed_visits, c, alpha, n_legal)
            outside = rank > k
            if root_override:
                # Bypasses widening: excluded from the primary denominator,
                # reported separately.
                self._forced_bypass[shape]["events"] += 1
                if outside:
                    self._forced_bypass[shape]["outside_events"] += 1
                continue
            for key in ("overall", _bucket(depth)):
                cell = self._cells[shape][key]
                cell["eligible_events"] += 1
                if outside:
                    cell["outside_events"] += 1
                    cell["excluded_prior_mass"] += float(parent.priors[selected_move])
                if first_touch:
                    cell["first_touch_events"] += 1
                    if outside:
                        cell["first_touch_outside_events"] += 1

    # -- output --------------------------------------------------------
    @staticmethod
    def _rate(num: int, den: int) -> Optional[float]:
        return None if den == 0 else num / den

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"by_shape": {}, "within_forced_events": self._within_forced_events}
        for shape, _c, _a in WIDENING_SHAPES:
            cells = {}
            for key, cell in self._cells[shape].items():
                cells[key] = dict(cell)
                cells[key]["outside_rate"] = self._rate(
                    cell["outside_events"], cell["eligible_events"])
                cells[key]["first_touch_outside_rate"] = self._rate(
                    cell["first_touch_outside_events"], cell["first_touch_events"])
            ft_rate = cells["overall"]["first_touch_outside_rate"]
            bypass = dict(self._forced_bypass[shape])
            out["by_shape"][shape] = {
                **cells,
                "forced_root_bypass_events": bypass["events"],
                "forced_root_bypass_outside_events": bypass["outside_events"],
                "forced_root_bypass_outside_rate": self._rate(
                    bypass["outside_events"], bypass["events"]),
                # None, never False, when the denominator is empty.
                "meaningfully_affected": (
                    None if ft_rate is None
                    else ft_rate >= MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE
                ),
            }
        return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_selection_tracer.py -v -p no:cacheprovider`
Expected: PASS — 18 passed (the two `k_of_n` tests are parametrized over 5 cases each; one extra test covers the outside-K counting path).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/selection_tracer.py tests/test_selection_tracer.py
git commit -m "feat(atlas-s1): selection tracer with per-search rank cache and online counters"
```

---

### Task 5: Identity qualification — each hook alone and all hooks together

**Files:**
- Modify: `tests/test_atlas_observer_identity.py`

**Interfaces:**
- Consumes: everything from Tasks 0–4.
- Produces: the §9 identity qualification.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_atlas_observer_identity.py
from scripts.GPU.alphazero.selection_tracer import SelectionTracer
from tests.test_atlas_observer_hooks import RecordingFlush, RecordingSelection


def test_each_hook_individually_preserves_identity():
    baseline = json.loads(GOLDEN.read_text())
    assert run_fixed_search(flush_observer=RecordingFlush()) == baseline
    assert run_fixed_search(selection_observer=RecordingSelection()) == baseline
    assert run_fixed_search(selection_observer=SelectionTracer()) == baseline


def test_all_hooks_together_preserve_identity():
    assert run_fixed_search(
        flush_observer=RecordingFlush(),
        selection_observer=SelectionTracer(),
    ) == json.loads(GOLDEN.read_text())


def test_hooks_actually_fired_during_the_identity_run():
    """Identity is vacuous if the hooks never ran."""
    fo, so = RecordingFlush(), RecordingSelection()
    run_fixed_search(flush_observer=fo, selection_observer=so)
    assert fo.calls and so.calls
```

- [ ] **Step 2: Run test to verify it passes or reveals a real perturbation**

Run: `.venv/bin/python -m pytest tests/test_atlas_observer_identity.py -v -p no:cacheprovider`
Expected: PASS — 6 passed.

If an identity test fails, the hook is perturbing search — almost certainly by mutating a node or advancing the RNG. **Fix the hook. Never weaken the comparison.**

- [ ] **Step 3: Prove byte-identical off across the whole suite**

Run: `.venv/bin/python -m pytest -p no:cacheprovider -q`
Expected: every pre-existing test passes unchanged. A single pre-existing failure means the default path changed; fix before proceeding.

- [ ] **Step 4: Commit**

```bash
git add tests/test_atlas_observer_identity.py
git commit -m "test(atlas-s1): identity qualification per-hook and all-on, non-vacuous"
```

---

### Task 6: Timing smoke and the producer no-go

**Files:**
- Modify: `tests/test_atlas_observer_identity.py`

**Interfaces:**
- Consumes: Tasks 4–5.
- Produces: a measured tracer-overhead figure — Stage 1's handoff artifact.

§8 requires a real 400-simulation timing smoke. If projected runtime breaks the atlas budget, **stop with an operational producer no-go — never silently reduce scope.**

- [ ] **Step 1: Write the timing smoke**

```python
# append to tests/test_atlas_observer_identity.py
import time


def test_tracer_overhead_is_measured_and_reported():
    """Not a pass/fail bar -- a MEASUREMENT. The atlas-budget decision is the
    operator's, made against this number."""
    def timed(**kw):
        # 400, not the golden's 200: section 8 requires a real 400-simulation smoke.
        t0 = time.perf_counter()
        run_fixed_search(n_simulations=400, **kw)
        return time.perf_counter() - t0

    off = min(timed() for _ in range(3))
    on = min(timed(flush_observer=RecordingFlush(),
                   selection_observer=SelectionTracer()) for _ in range(3))
    overhead = (on - off) / off
    print(f"\nATLAS STAGE 1 TRACER OVERHEAD: {overhead:+.1%} "
          f"(off {off:.3f}s, all-on {on:.3f}s)")
    assert on > 0 and off > 0
```

- [ ] **Step 2: Run it and record the number**

Run: `.venv/bin/python -m pytest tests/test_atlas_observer_identity.py::test_tracer_overhead_is_measured_and_reported -v -s -p no:cacheprovider`
Expected: PASS, printing the overhead. **Record that figure in the Stage 1 completion note** — it is the input to Stage 3's runtime estimate.

This uses `FakeEvaluator`, so it isolates tracer cost from NN cost and **overstates** the relative overhead compared with a real evaluator, where NN inference dominates. Report it as an upper bound, not as the production figure.

- [ ] **Step 3: Full suite, then commit**

```bash
.venv/bin/python -m pytest -p no:cacheprovider -q
git add tests/test_atlas_observer_identity.py
git commit -m "test(atlas-s1): tracer overhead timing smoke, reported as an upper bound"
```

---

## Stage 1 completion criteria

- [ ] Task 0 golden captured from **unmodified** `mcts.py`, and still reproducing.
- [ ] `on_flush_complete` fires at all three sites, after the clears, with types matching the flush counters.
- [ ] `on_select_child` fires in both descent loops, and `root_override=True` is proven reachable.
- [ ] Tracer counters accumulate online; cache clears on demand; zero denominators yield `None`.
- [ ] Identity holds per-hook and all-on, non-vacuously.
- [ ] Whole pre-existing suite passes unchanged.
- [ ] Tracer overhead measured and recorded.

## Out of scope

No corpus generation, no reservoir, no replay, no ladder, no read-out logic, no analysis, no artifact schema, no measurement run. Stage 2 is planned only after this stage's interfaces exist and qualify.

## Handoff to Stage 3 — one thing Stage 1 cannot prove

`N_actual = root.visit_count − I` is the §4 boundary quantity, and **Stage 1 never
exercises it with a nonzero `I`.** Every search here starts from a fresh
`MCTSNode(state=state)`, so `I = 0` throughout and the subtraction is trivially
correct in a way that establishes nothing.

Stage 3 owns the boundary consumer and **must pin the subtraction against a genuinely
nonzero `I`** — a root carrying inherited visits from a real `advance_root`, where
`root.visit_count` at flush time is `I + completed_target_backups` and the assertion
`320 ≤ N_actual ≤ 400` has real content. Until then, treat the derivation as designed
but unverified.

Two related Stage 3 obligations follow from the same gap: `SelectionTracer.clear_node_cache()`
is exercised here only by a direct unit call, never by an actual `advance_root`, and
`within_forced_simulation` is only observed on synchronous forced simulations, never
inside a warm-root replay.
