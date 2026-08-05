# Atlas Stage 4 — Read-outs A / B / C and Artifact Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and qualify labelling, capacity sizing, all three read-outs, and the artifact schema — every part pure and testable on synthetic ladder output, with no reservoir, checkpoint, MLX or measurement.

**Architecture:** Four pure modules over Stage 3's `LegResult` / `BoundaryRecord` / tracer snapshots: `atlas_labelling.py` (stable reference, classes, capacity), `atlas_readout_a.py` (boundary features, ridge classifier, deployability), `atlas_readout_b.py` (four-rung gate calibration), `atlas_readout_c.py` (retention, strata, shape selection). One `atlas_artifact.py` handles schema, provenance and `_jsonable` boundaries. Every input is a plain dataclass or dict, so the whole stage qualifies on synthetic rows.

**Tech Stack:** Python 3, stdlib only (`math`, `statistics`, `random` for the frozen bootstrap seed). **No scipy, no numpy-dependent fitting** — the ridge classifier is a few lines of closed-form gradient descent. Tests: `.venv/bin/python -m pytest -p no:cacheprovider`.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-03-convergence-atlas-design.md` §3, §5, §6, §7, §8. **§3–§12 are EXECUTION-FROZEN.** No parameter, threshold or predicate may be changed.
- **No reservoir generation, no checkpoint loading, no MLX execution, no measurement run.** Every test uses synthetic `LegResult` rows and synthetic tracer snapshots.
- **No `mcts.py` change.** Stage 1's scoped exception already delivered every hook.
- **Fail closed.** Insufficient classes, missing strata, absent snapshots or an undefined denominator are **stop** conditions with a named verdict — never a silent default, never an imputed value.
- **Undefined statistics are `None`, never `0`, never `false`.** This applies to every rate, median, quartile and boolean gate result.
- **`_jsonable` from `build_atlas_corpus` at every artifact boundary.** Do not re-key the geometry or ladder modules.
- **The three distribution gaps are operator/pilot measurements**, not values to infer from `FakeEvaluator` tests: real-scale throughput, the `remaining` distribution, and the inheritance-reset rate. Tests may exercise the *code paths* that consume them; they may not assert what the real values will be.
- Commit after every task.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/GPU/alphazero/atlas_labelling.py` (create) | Stable reference, misleading / stable-negative / ambiguous, class counts, capacity sizing, fail-closed verdicts. |
| `scripts/GPU/alphazero/atlas_readout_a.py` (create) | Boundary feature collector, ridge classifier, validation bars, deployability aggregation. |
| `scripts/GPU/alphazero/atlas_readout_b.py` (create) | Four-rung historical metrics, frozen convergence predicate, eligible-trigger gate. |
| `scripts/GPU/alphazero/atlas_readout_c.py` (create) | Two-snapshot consumption, static retention, strata, lexicographic shape selection, lag bound. |
| `scripts/GPU/alphazero/atlas_artifact.py` (create) | Row/run schema, provenance validation, `_jsonable` emission. |
| `tests/test_atlas_labelling.py`, `..._readout_a.py`, `..._readout_b.py`, `..._readout_c.py`, `..._artifact.py` (create) | One suite per module, synthetic inputs only. |

---

### Task 0: Producer closure — freeze what the pure analyses will consume

**Files:**
- Modify: `scripts/GPU/alphazero/warm_prefix_replay.py`
- Modify: `scripts/GPU/alphazero/selection_tracer.py`
- Test: `tests/test_atlas_producer_closure.py`

**Interfaces:**
- Produces: `capture_tree_state(root) -> dict` — the **complete compact capture schema** below; `check_backup_invariant(...)`; `reference_line_summary(root, legs) -> dict`; `SelectionTracer` extended with **simultaneous `K(n+14)` counters**; `run_additive_ladder` returning `captures = {at_start, at_boundary, at_400}` and `reference_lines`.

**The capture schema, frozen here because every consumer depends on it.** One
dict, taken at one instant, carrying everything the pure analyses need after the
tree has moved on:

```python
{
  # section 6a backup accounting
  "D3": int,                       # sum of visits at depth EXACTLY 3
  # root visit distribution -- section 6 leader margin, section 7 metrics
  "root_visits": int,              # I + backups so far  (== K(n)'s effective n)
  "total_child_visits": int,
  "top_child_visits": int | None,
  "second_child_visits": int | None,
  "n_visited_children": int,
  "one_visit_children": int,
  # section 6 leader breadth: children OF the canonical leader, not of the root
  "leader_move": int | None,
  "leader_breadth": int | None,
  # section 6 normalized policy entropy, H / log(n_legal)
  "policy_entropy": float | None,
  "n_legal": int,
}
```

`root_visits` is the field §6a's amendment requires for `K(n)`: at a warm root the
effective `n` is `I + N_actual`, never the nominal 320. Capturing it here is what
makes that computable later.

`leader_breadth` counts children **of the canonical `visit_leader_move`**, not of
the root — a distinct quantity, and the chain test previously conflated them.

> **Why this task exists, and why it is first.** Stage 4's analyses are pure, which
> made three producer gaps invisible until review:
>
> 1. `depth3plus_backup_fraction` was read off **selection events**, but those are edge
>    traversals — a depth-5 simulation emits five of them and is one backup. §6a now
>    freezes the two-point `D3` accounting instead.
> 2. Boundary features were computed from `pre.root` **after** the ladder, but the
>    ladder mutates that root in place through all four legs, so the values described
>    the 6,400 tree. They must be frozen at the boundary and at `B = 400`.
> 3. The `K(n+14)` lagged count was **caller-supplied**. No real row could produce it,
>    so the bound was untestable in practice.
>
> Building the pure analyses on top of these would have produced confidently wrong
> numbers with green tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_producer_closure.py
import random

import pytest

from scripts.GPU.alphazero.corpus_geometry import GameMeta
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.selection_tracer import SelectionTracer
from scripts.GPU.alphazero.warm_prefix_replay import (
    BatchSafeBoundaryObserver, capture_tree_state, replay_prefix,
    run_additive_ladder,
)

from tests.eval_fakes import FakeEvaluator

BASE = 20400000
SIZE = 6


class _N:
    """Stand-in reaching the WHOLE capture, not just the D3 walker.

    capture_tree_state calls visit_leader_move(root), which reads each child's
    `.move`, and reads root.priors. A stand-in without them raises
    AttributeError before D3 is ever checked.
    """
    def __init__(self, visits, kids=None, move=0, priors=None):
        self.visit_count = visits
        self.children = kids or {}
        self.move = move
        self.priors = priors if priors is not None else {
            k: 1.0 / (i + 1) for i, k in enumerate(sorted(self.children))
        }


def test_D3_counts_nodes_at_depth_exactly_three():
    """Every backup reaching depth >=3 passes through exactly ONE depth-3 node,
    so summing visits there counts each such backup once."""
    root = _N(10, {0: _N(6, {0: _N(4, {0: _N(3, move=0), 1: _N(1, move=1)},
                                    move=0)}, move=0),
                   1: _N(4, {0: _N(2, {0: _N(2, move=0)}, move=0)}, move=1)})
    acc = capture_tree_state(root)
    assert acc["D3"] == 3 + 1 + 2          # depth-3 nodes only
    assert acc["n_visited_children"] == 2
    assert acc["one_visit_children"] == 0


def test_one_visit_children_is_counted_on_the_root_only():
    root = _N(5, {0: _N(3, move=0), 1: _N(1, move=1),
                  2: _N(1, move=2), 3: _N(0, move=3)})
    acc = capture_tree_state(root)
    assert acc["one_visit_children"] == 2
    assert acc["n_visited_children"] == 3        # the 0-visit child is excluded


def test_tracer_counts_the_lagged_bound_simultaneously():
    """Section 6a: K(n+14) is PRODUCED online, never supplied by a caller."""
    t = SelectionTracer()
    parent = type("P", (), {"priors": {i: 1.0 / (i + 1) for i in range(40)},
                            "children": {}})()
    rank20 = sorted(parent.priors.items(), key=lambda kv: (-kv[1], kv[0]))[19][0]
    t.on_select_child(parent=parent, selected_move=rank20, existing_child=None,
                      depth=1, parent_completed_visits=5, root_override=False,
                      within_forced_simulation=False)
    snap = t.snapshot()["by_shape"]["c4a05"]["overall"]
    # K(5) = 9 so rank 20 is outside; K(19) = 18 so it is still outside.
    assert snap["first_touch_outside_events"] == 1
    assert "lagged_first_touch_outside_events" in snap
    assert snap["lagged_first_touch_outside_events"] == 1


def test_the_lagged_counter_is_never_larger_than_the_unlagged_one():
    """K(n+14) >= K(n), so the lagged admitted set is wider and can only
    exclude fewer events."""
    t = SelectionTracer()
    parent = type("P", (), {"priors": {i: 1.0 / (i + 1) for i in range(60)},
                            "children": {}})()
    for rank in range(1, 40):
        mv = sorted(parent.priors.items(), key=lambda kv: (-kv[1], kv[0]))[rank - 1][0]
        t.on_select_child(parent=parent, selected_move=mv, existing_child=None,
                          depth=1, parent_completed_visits=5,
                          root_override=False, within_forced_simulation=False)
    o = t.snapshot()["by_shape"]["c4a05"]["overall"]
    assert o["lagged_first_touch_outside_events"] <= o["first_touch_outside_events"]


def _history(n, size=SIZE):
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    s = TwixtState(active_size=size, to_move="red")
    out = []
    for _ in range(n):
        lm = s.legal_moves()
        if not lm:
            break
        out.append(lm[0])
        s = s.apply_move(lm[0])
    return out


def test_features_are_frozen_at_the_boundary_not_after_the_ladder():
    """The decisive one: the root is mutated to 6,400 by leg 4, so a
    post-ladder read describes a different tree entirely."""
    hist = _history(4)
    meta = GameMeta(game_id=0, seed=BASE, n_moves=len(hist), start_player="red")
    m = MCTS(FakeEvaluator(value=0.0),
             MCTSConfig(n_simulations=1, eval_batch_size=14,
                        stall_flush_sims=48, pending_virtual_visits=8),
             random.Random(BASE))
    pre = replay_prefix(m, meta, hist, target_ply=2, active_size=SIZE)
    tracer = SelectionTracer()
    m._selection_observer = tracer
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I, threshold=40,
                                    leg_B=80, tracer=tracer)
    _legs, snaps = run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                                       boundary_observer=obs,
                                       target_tracer=tracer,
                                       increments=(80, 80, 80, 80))
    fb = snaps["captures"]["at_boundary"]
    f4 = snaps["captures"]["at_400"]
    assert fb is not None and f4 is not None

    # 1. The capture is anchored to the boundary instant, exactly.
    assert fb["root_visits"] == pre.inherited_I + obs.record.N_actual

    # 2. It must not have moved when later legs ran. Re-reading the returned
    #    dict after the ladder finished proves the snapshot was taken by value.
    assert snaps["captures"]["at_boundary"]["root_visits"] == fb["root_visits"]

    # 3. At least one field must have STRICTLY changed by the end, or `<=`
    #    comparisons would pass even if both reads happened at the end.
    after = capture_tree_state(pre.root)
    assert after["root_visits"] > fb["root_visits"]
    assert after["root_visits"] == pre.inherited_I + 320       # 4 x 80


def test_the_backup_invariant_is_asserted():
    """Section 6a: 0 <= D3(boundary) - D3(start) <= N_actual. A violation means
    the accounting is wrong and the row must fail, not be recorded."""
    from scripts.GPU.alphazero.warm_prefix_replay import check_backup_invariant
    assert check_backup_invariant(d3_start=10, d3_boundary=40, n_actual=326) is True
    with pytest.raises(ValueError):
        check_backup_invariant(d3_start=40, d3_boundary=10, n_actual=326)
    with pytest.raises(ValueError):
        check_backup_invariant(d3_start=0, d3_boundary=400, n_actual=326)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_producer_closure.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'capture_tree_state'`

- [ ] **Step 3: Implement**

In `warm_prefix_replay.py`:

```python
def capture_tree_state(root: MCTSNode) -> Dict[str, Any]:
    """The COMPLETE compact capture (section 6a), taken at ONE instant.

    D3 sums visit_count over nodes at depth EXACTLY 3 below `root`: every backup
    reaching depth >=3 passes through exactly one of them, so the sum counts each
    such backup once. Selection events cannot substitute -- they are edge
    traversals, and one deep simulation emits several.

    Everything else here exists because the ladder mutates this root in place:
    after leg 4 it describes 6,400 simulations, so any value not frozen now is
    unrecoverable.
    """
    d3 = 0
    frontier = [(root, 0)]
    while frontier:
        node, depth = frontier.pop()
        if depth == 3:
            d3 += node.visit_count
            continue                     # deeper nodes are already counted here
        if depth < 3:
            frontier.extend((c, depth + 1) for c in node.children.values())

    visited = [c for c in root.children.values() if c.visit_count > 0]
    counts = sorted((c.visit_count for c in visited), reverse=True)
    leader = visit_leader_move(root)
    leader_breadth = (len(root.children[leader].children)
                      if leader is not None and leader in root.children else None)

    priors = [p for p in (root.priors or {}).values() if p > 0]
    entropy = None
    if len(priors) >= 2:
        s = sum(priors)
        norm = [p / s for p in priors]
        # Normalized by log(n_legal), the existing convention.
        entropy = (-sum(q * math.log(q) for q in norm)) / math.log(len(root.priors))

    return {
        "D3": d3,
        # K(n)'s EFFECTIVE n at a warm root -- I + backups, not the nominal budget.
        "root_visits": root.visit_count,
        "total_child_visits": sum(counts),
        "top_child_visits": counts[0] if counts else None,
        "second_child_visits": counts[1] if len(counts) >= 2 else None,
        "n_visited_children": len(visited),
        "one_visit_children": sum(1 for c in visited if c.visit_count == 1),
        "leader_move": leader,
        # Children OF THE LEADER, not of the root -- a different quantity.
        "leader_breadth": leader_breadth,
        "policy_entropy": entropy,
        "n_legal": len(root.priors or {}),
    }


def reference_line_summary(root: MCTSNode, legs: Sequence[LegResult]
                           ) -> Dict[str, Any]:
    """What Read-out C needs from the 3,200 / 6,400 reference lines.

    Preserves the stable deep root move, its best reply, a two-ply horizon, and
    -- critically -- the PRIORS and EFFECTIVE parent visit counts at each of
    those nodes, since `static_retention` cannot recompute them once the tree has
    advanced.
    """
    d = {l.nominal_B: l for l in legs}
    deep_move = d[6400].selected_move
    out: Dict[str, Any] = {
        "stable_deep_move": deep_move,
        "root_priors": dict(root.priors or {}),
        "root_effective_visits": root.visit_count,
        "reply": None, "two_ply": None,
    }
    child = root.children.get(deep_move) if deep_move is not None else None
    if child is not None:
        reply = visit_leader_move(child)
        out["reply"] = {
            "move": reply,
            "priors": dict(child.priors or {}),
            "effective_visits": child.visit_count,
        }
        gc = child.children.get(reply) if reply is not None else None
        if gc is not None:
            out["two_ply"] = {
                "move": visit_leader_move(gc),
                "priors": dict(gc.priors or {}),
                "effective_visits": gc.visit_count,
            }
    return out


def check_backup_invariant(d3_start: int, d3_boundary: int,
                           n_actual: int) -> bool:
    """Section 6a. A violation is a broken accounting, not a datum."""
    delta = d3_boundary - d3_start
    if delta < 0 or delta > n_actual:
        raise ValueError(
            f"backup accounting invariant violated: D3 delta {delta} outside "
            f"[0, {n_actual}]; the row must fail rather than be recorded")
    return True
```

`run_additive_ladder` additionally captures `capture_tree_state(root)` at
the start of leg 1, hands it to the boundary observer so the boundary capture can
be frozen at the same instant as `at_boundary`, and freezes another immediately
after leg 1 as `features_at_400`. `snapshots` gains `features_at_start`,
`features_at_boundary`, `features_at_400`.

In `selection_tracer.py`, each cell gains `lagged_first_touch_outside_events`,
incremented in the same pass using `k_of_n(parent_completed_visits + 14, ...)`.
`BATCH_LAG = 14` is a module constant.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_producer_closure.py tests/test_warm_prefix_replay.py tests/test_selection_tracer.py -v -p no:cacheprovider`
Expected: PASS — 6 new, plus the existing 32 warm-replay and 18 tracer tests all still green.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/warm_prefix_replay.py scripts/GPU/alphazero/selection_tracer.py tests/test_atlas_producer_closure.py
git commit -m "feat(atlas-s4): producer closure -- backup accounting, frozen captures, lagged counters"
```

---

### Task 1: Labelling and capacity sizing

**Files:**
- Create: `scripts/GPU/alphazero/atlas_labelling.py`
- Test: `tests/test_atlas_labelling.py`

**Interfaces:**
- Consumes: Stage 3's `LegResult` (fields `nominal_B`, `root_value`, `selected_move`, `top_two_margin`).
- Produces: `stable_reference(legs) -> dict`; `classify_row(legs) -> str` in `{"misleading","stable_negative","ambiguous","no_stable_reference"}`; `class_counts(rows) -> dict`; `size_from_pilot(counts, pilot_n=24) -> dict`; `final_capacity_gate(counts) -> dict`.

Frozen §5 thresholds, none of them adjustable: stable reference needs 3,200/6,400 move
agreement, `|V3200 − V6400| ≤ 0.10`, and 6,400 top-two margin `≥ 0.05`. Misleading is an
**OR** (`|V400 − V6400| ≥ 0.25` or a different 400 move); stable-negative is an **AND**
(same move and `|V400 − V6400| ≤ 0.10`). Everything else is ambiguous, kept, and counted.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_labelling.py
import pytest

from scripts.GPU.alphazero.atlas_labelling import (
    class_counts, classify_row, final_capacity_gate, size_from_pilot,
    stable_reference,
)
from scripts.GPU.alphazero.warm_prefix_replay import LegResult


def _legs(v400, v1600, v3200, v6400, m400, m3200, m6400, margin=0.20):
    """Four rungs with only the fields labelling reads."""
    vals = (v400, v1600, v3200, v6400)
    moves = (m400, m400, m3200, m6400)
    out = []
    for i, (b, v, m) in enumerate(zip((400, 1600, 3200, 6400), vals, moves)):
        out.append(LegResult(
            nominal_B=b, inherited_I=10, effective=10 + b, root_value=v,
            selected_move=m, selected_move_prior_rank=1, top_share=0.5,
            top_two_margin=(margin if b == 6400 else 0.30),
            effective_children=12.0, n_visited_children=20,
            visit_counts={m: 100}))
    return out


def test_stable_reference_requires_all_three_conditions():
    ok = _legs(0.9, 0.5, 0.10, 0.05, m400=7, m3200=3, m6400=3)
    assert stable_reference(ok)["stable"] is True

    moves_disagree = _legs(0.9, 0.5, 0.10, 0.05, m400=7, m3200=3, m6400=9)
    assert stable_reference(moves_disagree)["stable"] is False

    values_apart = _legs(0.9, 0.5, 0.90, 0.05, m400=7, m3200=3, m6400=3)
    assert stable_reference(values_apart)["stable"] is False

    thin_margin = _legs(0.9, 0.5, 0.10, 0.05, m400=7, m3200=3, m6400=3, margin=0.01)
    assert stable_reference(thin_margin)["stable"] is False


def test_misleading_is_an_OR_of_value_and_move():
    by_value = _legs(0.9, 0.5, 0.10, 0.05, m400=3, m3200=3, m6400=3)
    assert classify_row(by_value) == "misleading"       # |0.9-0.05| >= 0.25
    by_move = _legs(0.06, 0.05, 0.05, 0.05, m400=7, m3200=3, m6400=3)
    assert classify_row(by_move) == "misleading"        # different 400 move


def test_stable_negative_is_an_AND():
    r = _legs(0.06, 0.05, 0.05, 0.05, m400=3, m3200=3, m6400=3)
    assert classify_row(r) == "stable_negative"


def test_the_ambiguous_band_is_kept_not_forced():
    """Same move, value gap in (0.10, 0.25) -- neither class."""
    r = _legs(0.20, 0.10, 0.05, 0.05, m400=3, m3200=3, m6400=3)
    assert classify_row(r) == "ambiguous"


def test_rows_without_a_stable_reference_are_their_own_class():
    r = _legs(0.9, 0.5, 0.10, 0.05, m400=7, m3200=3, m6400=9)
    assert classify_row(r) == "no_stable_reference"


def test_class_counts_report_components_separately():
    rows = [_legs(0.9, 0.5, 0.10, 0.05, 3, 3, 3),          # misleading by value
            _legs(0.06, 0.05, 0.05, 0.05, 7, 3, 3),        # misleading by move
            _legs(0.06, 0.05, 0.05, 0.05, 3, 3, 3)]        # stable negative
    c = class_counts(rows)
    assert c["misleading"] == 2 and c["stable_negative"] == 1
    # Section 5: value and move components reported separately -- a detector that
    # predicts value correction but not move error is weaker evidence.
    assert c["misleading_by_value"] == 1 and c["misleading_by_move"] == 1


def test_sizing_matches_the_frozen_formula():
    counts = {"misleading": 8, "stable_negative": 9}       # of 24 pilot rows
    r = size_from_pilot(counts)
    # N_required = max(24/(0.4*p_m), 30/(0.4*p_s)); rounded up to a multiple of 40
    assert r["p_m"] == pytest.approx(8 / 24)
    assert r["p_s"] == pytest.approx(9 / 24)
    assert r["N"] % 40 == 0 and 200 <= r["N"] <= 400
    assert r["verdict"] == "OK"


def test_sizing_fails_closed_on_a_zero_class_frequency():
    r = size_from_pilot({"misleading": 0, "stable_negative": 9})
    assert r["verdict"] == "PROJECTED_CAPACITY_NO_GO"
    assert r["N"] is None                # None, never a defaulted number


def test_sizing_fails_closed_when_the_requirement_exceeds_400():
    r = size_from_pilot({"misleading": 1, "stable_negative": 1})
    assert r["verdict"] == "PROJECTED_CAPACITY_NO_GO"


def test_final_capacity_gate_needs_20_misleading_and_25_stable_negative():
    assert final_capacity_gate({"misleading": 20, "stable_negative": 25})["verdict"] == "OK"
    short = final_capacity_gate({"misleading": 19, "stable_negative": 25})
    assert short["verdict"] == "CAPACITY_FAILURE"
    assert "misleading" in short["short_of"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_labelling.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named '...atlas_labelling'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/atlas_labelling.py
"""Atlas labelling and capacity sizing -- design sections 5 and 3, FROZEN.

Pure: consumes Stage 3 LegResult rows and nothing else, so the whole stage
qualifies on synthetic input with no reservoir and no GPU.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

# Frozen section 5 thresholds.
STABLE_VALUE_TOL = 0.10          # |V3200 - V6400|
STABLE_TOP_TWO_MARGIN = 0.05     # normalized, at 6400
MISLEADING_VALUE_GAP = 0.25      # |V400 - V6400|
STABLE_NEGATIVE_VALUE_TOL = 0.10

# Frozen section 3 sizing.
PILOT_GAMES = 24
DISCOVERY_FRACTION = 0.6
VALIDATION_FRACTION = 0.4
MARGIN = 1.20
MIN_VALIDATION_MISLEADING = 20
MIN_VALIDATION_STABLE_NEGATIVE = 25
ALLOWED_N = (200, 240, 280, 320, 360, 400)


def _by_b(legs: Sequence[Any]) -> Dict[int, Any]:
    d = {l.nominal_B: l for l in legs}
    missing = {400, 1600, 3200, 6400} - set(d)
    if missing:
        raise ValueError(f"missing rungs {sorted(missing)}; all four are required")
    return d


def stable_reference(legs: Sequence[Any]) -> Dict[str, Any]:
    """Section 5: 3,200 and 6,400 agree on the move, their values are within
    0.10, and the 6,400 top-two margin is at least 0.05.

    Without the 3,200 rung there is no stability check at all and "6,400 is
    truth" stops being falsifiable -- which is why the ladder carries it.
    """
    d = _by_b(legs)
    moves_agree = d[3200].selected_move == d[6400].selected_move
    value_close = abs(d[3200].root_value - d[6400].root_value) <= STABLE_VALUE_TOL
    margin = d[6400].top_two_margin
    margin_ok = margin is not None and margin >= STABLE_TOP_TWO_MARGIN
    return {
        "stable": bool(moves_agree and value_close and margin_ok),
        "moves_agree": moves_agree, "value_close": value_close,
        "margin_ok": margin_ok, "top_two_margin": margin,
        "stable_deep_move": d[6400].selected_move if moves_agree else None,
    }


def classify_row(legs: Sequence[Any]) -> str:
    d = _by_b(legs)
    ref = stable_reference(legs)
    if not ref["stable"]:
        return "no_stable_reference"
    deep = ref["stable_deep_move"]
    value_gap = abs(d[400].root_value - d[6400].root_value)
    same_move = d[400].selected_move == deep
    # Misleading is an OR; stable-negative is an AND. The asymmetry is
    # deliberate and the ambiguous band between them is kept, not forced.
    if (value_gap >= MISLEADING_VALUE_GAP) or (not same_move):
        return "misleading"
    if same_move and value_gap <= STABLE_NEGATIVE_VALUE_TOL:
        return "stable_negative"
    return "ambiguous"


def class_counts(rows: Sequence[Sequence[Any]]) -> Dict[str, int]:
    """Counts, with the misleading components reported separately (section 5)."""
    c = {"misleading": 0, "stable_negative": 0, "ambiguous": 0,
         "no_stable_reference": 0, "misleading_by_value": 0,
         "misleading_by_move": 0}
    for legs in rows:
        label = classify_row(legs)
        c[label] += 1
        if label == "misleading":
            d = _by_b(legs)
            deep = stable_reference(legs)["stable_deep_move"]
            if abs(d[400].root_value - d[6400].root_value) >= MISLEADING_VALUE_GAP:
                c["misleading_by_value"] += 1
            if d[400].selected_move != deep:
                c["misleading_by_move"] += 1
    return c


def _round_up_40(x: float) -> int:
    return int(math.ceil(x / 40.0) * 40)


def size_from_pilot(counts: Dict[str, int], pilot_n: int = PILOT_GAMES
                    ) -> Dict[str, Any]:
    """Section 3's frozen staged sizing. Fails CLOSED on a zero frequency or a
    requirement above 400 -- never defaults a number."""
    p_m = counts.get("misleading", 0) / pilot_n
    p_s = counts.get("stable_negative", 0) / pilot_n
    base = {"p_m": p_m, "p_s": p_s}
    if p_m == 0 or p_s == 0:
        return {**base, "verdict": "PROJECTED_CAPACITY_NO_GO", "N": None,
                "reason": "a pilot class frequency is zero; no N can satisfy it"}
    need = max(MARGIN * MIN_VALIDATION_MISLEADING / (VALIDATION_FRACTION * p_m),
               MARGIN * MIN_VALIDATION_STABLE_NEGATIVE / (VALIDATION_FRACTION * p_s))
    n = _round_up_40(need)
    if n > max(ALLOWED_N):
        return {**base, "verdict": "PROJECTED_CAPACITY_NO_GO", "N": None,
                "required": n,
                "reason": f"required N {n} exceeds the frozen maximum {max(ALLOWED_N)}"}
    return {**base, "verdict": "OK", "N": max(n, min(ALLOWED_N)), "required": n}


def final_capacity_gate(counts: Dict[str, int]) -> Dict[str, Any]:
    """Section 3: the completed VALIDATION split must hold >=20 misleading and
    >=25 stable-negative. Otherwise the atlas ends as an operational capacity
    failure -- do not weaken labels, move ambiguous rows, or add games."""
    short = []
    if counts.get("misleading", 0) < MIN_VALIDATION_MISLEADING:
        short.append("misleading")
    if counts.get("stable_negative", 0) < MIN_VALIDATION_STABLE_NEGATIVE:
        short.append("stable_negative")
    return {"verdict": "CAPACITY_FAILURE" if short else "OK",
            "short_of": short, "counts": dict(counts)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_labelling.py -v -p no:cacheprovider`
Expected: PASS — 10 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_labelling.py tests/test_atlas_labelling.py
git commit -m "feat(atlas-s4): labelling, class counts and frozen capacity sizing"
```

---

### Task 2: Read-out A — features from the frozen captures

**Files:**
- Create: `scripts/GPU/alphazero/atlas_readout_a.py`
- Test: `tests/test_atlas_readout_a.py`

**Interfaces:**
- Consumes: Task 0's capture dicts and `N_actual`. **Never a live root, never a selection-tracer snapshot.**
- Produces: `FEATURE_NAMES`; `collect_features(capture_start, capture_boundary, n_actual) -> dict`.

> Two things this task must get right, both of which the pre-amendment version got
> wrong. The depth feature comes from §6a's **two-point `D3` accounting**, not from
> selection events — those are edge traversals and one deep simulation emits several.
> And the capture is a **frozen dict**: by the time Read-out A runs, the live root has
> advanced to 6,400.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_readout_a.py
import math

import pytest

from scripts.GPU.alphazero.atlas_readout_a import FEATURE_NAMES, collect_features


def _cap(D3=0, root_visits=400, total=390, top=200, second=100, n_vis=30,
         one_vis=5, leader=7, breadth=17, entropy=0.8, n_legal=500):
    return {"D3": D3, "root_visits": root_visits, "total_child_visits": total,
            "top_child_visits": top, "second_child_visits": second,
            "n_visited_children": n_vis, "one_visit_children": one_vis,
            "leader_move": leader, "leader_breadth": breadth,
            "policy_entropy": entropy, "n_legal": n_legal}


def test_exactly_five_frozen_features():
    assert len(FEATURE_NAMES) == 5
    assert set(FEATURE_NAMES) == {
        "one_visit_backup_share", "depth3plus_backup_fraction",
        "leader_visit_margin", "root_policy_entropy", "leader_breadth"}


def test_depth_feature_uses_the_two_point_D3_accounting():
    """(D3(boundary) - D3(start)) / N_actual -- NOT selection events."""
    f = collect_features(_cap(D3=40), _cap(D3=140), n_actual=326)
    assert f["depth3plus_backup_fraction"] == pytest.approx(100 / 326)


def test_the_backup_invariant_is_enforced_here_too():
    with pytest.raises(ValueError):
        collect_features(_cap(D3=140), _cap(D3=40), n_actual=326)      # negative
    with pytest.raises(ValueError):
        collect_features(_cap(D3=0), _cap(D3=999), n_actual=326)       # > N_actual


def test_remaining_features_come_from_the_boundary_capture():
    f = collect_features(_cap(D3=0), _cap(D3=10, one_vis=6, n_vis=30, top=200,
                                          second=100, total=400, entropy=0.77,
                                          breadth=17), n_actual=326)
    assert f["one_visit_backup_share"] == pytest.approx(6 / 30)
    assert f["leader_visit_margin"] == pytest.approx((200 - 100) / 400)
    assert f["root_policy_entropy"] == pytest.approx(0.77)
    assert f["leader_breadth"] == 17


def test_undefined_features_are_None_not_zero():
    f = collect_features(
        _cap(D3=0),
        _cap(D3=0, n_vis=0, one_vis=0, top=None, second=None, total=0,
             entropy=None, breadth=None),
        n_actual=326)
    for k in ("one_visit_backup_share", "leader_visit_margin",
              "root_policy_entropy", "leader_breadth"):
        assert f[k] is None, f"{k} must be None when undefined"


def test_a_single_visited_child_has_no_margin():
    f = collect_features(_cap(D3=0), _cap(D3=0, n_vis=1, second=None),
                         n_actual=326)
    assert f["leader_visit_margin"] is None


def test_zero_N_actual_yields_None_not_a_division_error():
    f = collect_features(_cap(D3=0), _cap(D3=0), n_actual=0)
    assert f["depth3plus_backup_fraction"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_a.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/atlas_readout_a.py
"""Atlas Read-out A -- design section 6 and amendment 6a, FROZEN.

Consumes Task 0's FROZEN CAPTURES, never a live root: the ladder mutates that
root through all four legs, so a later read describes the 6,400 tree.

Pure: every input is a plain dict, so this qualifies on synthetic rows.
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

FEATURE_NAMES: Tuple[str, ...] = (
    "one_visit_backup_share",
    "depth3plus_backup_fraction",
    "leader_visit_margin",
    "root_policy_entropy",
    "leader_breadth",
)


def collect_features(capture_start: Dict[str, Any],
                     capture_boundary: Dict[str, Any],
                     n_actual: int) -> Dict[str, Optional[float]]:
    """The five frozen features. Undefined -> None, never 0.0."""
    delta = capture_boundary["D3"] - capture_start["D3"]
    if delta < 0 or delta > max(n_actual, 0):
        raise ValueError(
            f"backup accounting invariant violated: D3 delta {delta} outside "
            f"[0, {n_actual}]; the row must fail rather than be recorded")

    n_vis = capture_boundary["n_visited_children"]
    top = capture_boundary["top_child_visits"]
    second = capture_boundary["second_child_visits"]
    total = capture_boundary["total_child_visits"]
    return {
        "one_visit_backup_share": ((capture_boundary["one_visit_children"] / n_vis)
                                   if n_vis else None),
        "depth3plus_backup_fraction": ((delta / n_actual) if n_actual else None),
        "leader_visit_margin": (((top - second) / total)
                                if (top is not None and second is not None
                                    and total) else None),
        "root_policy_entropy": capture_boundary["policy_entropy"],
        "leader_breadth": capture_boundary["leader_breadth"],
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_a.py -v -p no:cacheprovider`
Expected: PASS — 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_readout_a.py tests/test_atlas_readout_a.py
git commit -m "feat(atlas-s4): Read-out A features from the frozen two-point captures"
```

---

### Task 3: Read-out A — classifier, string labels, bars, deployability

**Files:**
- Modify: `scripts/GPU/alphazero/atlas_readout_a.py`
- Test: `tests/test_atlas_readout_a.py`

**Interfaces:**
- Produces: `LABEL_TO_Y`; `prepare_rows(rows) -> dict`; `standardize(...)`; `fit_ridge_logistic(...)`; `auc(...)`; `bootstrap_auc_lower_bound(...)`; `evaluate_detector(discovery, validation) -> dict`; `deployability(...)`.

> **`classify_row` returns STRINGS.** A detector expecting numeric `1`/`0` would count
> every real row as neither class and silently train on nothing. `prepare_rows` filters
> to the two eligible classes, maps them explicitly, **rejects** rows with a missing
> feature per §6a, and reports the rejection count — and capacity is rechecked **after**
> rejection, because rejection is what can push a split below its own gate.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_atlas_readout_a.py
from scripts.GPU.alphazero.atlas_readout_a import (
    LABEL_TO_Y, auc, bootstrap_auc_lower_bound, deployability,
    evaluate_detector, fit_ridge_logistic, prepare_rows, standardize,
)


def _row(label, **feats):
    base = {k: 0.5 for k in FEATURE_NAMES}
    base.update(feats)
    return {"label": label, "features": base}


def test_string_labels_map_explicitly_and_other_classes_are_dropped():
    assert LABEL_TO_Y == {"misleading": 1, "stable_negative": 0}
    r = prepare_rows([_row("misleading"), _row("stable_negative"),
                      _row("ambiguous"), _row("no_stable_reference")])
    assert r["y"] == [1, 0]
    assert r["dropped_ineligible"] == 2


def test_rows_with_a_missing_feature_are_REJECTED_and_reported():
    bad = _row("misleading"); bad["features"]["leader_breadth"] = None
    r = prepare_rows([bad, _row("stable_negative")])
    assert r["rejected_missing_features"] == 1
    assert len(r["y"]) == 1


def test_capacity_is_rechecked_AFTER_rejection():
    """Rejection is exactly what can push a split below its own gate."""
    rows = [_row("misleading") for _ in range(20)] + \
           [_row("stable_negative") for _ in range(25)]
    rows[0]["features"]["root_policy_entropy"] = None       # one rejection
    r = evaluate_detector(discovery=rows, validation=rows)
    assert r["verdict"] == "INSUFFICIENT_CLASSES"
    assert r["n_misleading"] == 19


def test_fitting_requires_both_DISCOVERY_classes():
    disc = [_row("misleading") for _ in range(30)]           # one class only
    val = [_row("misleading") for _ in range(20)] + \
          [_row("stable_negative") for _ in range(25)]
    r = evaluate_detector(discovery=disc, validation=val)
    assert r["verdict"] == "INSUFFICIENT_DISCOVERY_CLASSES"


def test_auc_is_one_for_perfect_separation_and_half_for_none():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == pytest.approx(0.5)


def test_auc_is_None_when_a_class_is_absent():
    assert auc([0.1, 0.2], [0, 0]) is None       # None, never a defaulted 0.5


def test_standardization_stats_come_from_discovery_only():
    disc = [{"a": 1.0}, {"a": 3.0}]
    _z, stats = standardize(disc, feature_names=("a",))
    z_val, stats2 = standardize([{"a": 5.0}], feature_names=("a",), stats=stats)
    assert stats2 == stats
    assert z_val[0][0] == pytest.approx((5.0 - 2.0) / stats["a"][1])


def test_standardize_rejects_a_missing_feature_rather_than_imputing():
    with pytest.raises(ValueError, match="missing features"):
        standardize([{"a": None}], feature_names=("a",))


def test_ridge_logistic_separates_a_linearly_separable_set():
    X, y = [[-2.0], [-1.0], [1.0], [2.0]], [0, 0, 1, 1]
    model = fit_ridge_logistic(X, y)
    assert auc([model["predict"](x) for x in X], y) == pytest.approx(1.0)


def test_deployability_fails_when_the_MEDIAN_remaining_is_zero():
    r = deployability([0, 0, 0, 40, 60])
    assert r["median_remaining"] == 0 and r["verdict"] == "NOT_DEPLOYABLE"
    assert r["zero_budget_fraction"] == pytest.approx(3 / 5)


def test_deployability_passes_with_a_positive_median_and_reports_quartiles():
    r = deployability([10, 40, 60, 70, 80])
    assert r["verdict"] == "DEPLOYABLE" and len(r["quartiles"]) == 3


def test_deployability_reports_strata_without_gating_on_them():
    r = deployability([0, 40, 60], strata={"late": [0, 0], "midgame": [60]})
    assert set(r["by_stratum"]) == {"late", "midgame"}
    assert "verdict" not in r["by_stratum"]["late"]


def test_deployability_of_an_empty_set_is_None_not_zero():
    r = deployability([])
    assert r["median_remaining"] is None and r["zero_budget_fraction"] is None
    assert r["verdict"] == "NO_ROWS"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_a.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'LABEL_TO_Y'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/GPU/alphazero/atlas_readout_a.py
LABEL_TO_Y = {"misleading": 1, "stable_negative": 0}

MIN_VALIDATION_MISLEADING = 20
MIN_VALIDATION_STABLE_NEGATIVE = 25
AUC_BAR = 0.75
AUC_LOWER_BOUND_BAR = 0.60
MAX_FLAG_RATE = 0.25
MIN_PRECISION = 0.60
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260804


def prepare_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """String labels -> numeric, ineligible classes dropped, missing-feature
    rows REJECTED and counted (section 6a)."""
    feats, y = [], []
    dropped = rejected = 0
    for r in rows:
        if r["label"] not in LABEL_TO_Y:
            dropped += 1                       # ambiguous / no_stable_reference
            continue
        f = r["features"]
        if any(f.get(k) is None for k in FEATURE_NAMES):
            rejected += 1
            continue
        feats.append(f)
        y.append(LABEL_TO_Y[r["label"]])
    return {"features": feats, "y": y, "dropped_ineligible": dropped,
            "rejected_missing_features": rejected}


def standardize(rows, feature_names: Sequence[str] = FEATURE_NAMES,
                stats: Optional[Dict[str, Tuple[float, float]]] = None):
    """Z-score; `stats` learned on DISCOVERY only. A missing feature raises --
    imputing the mean would fabricate a maximally-uninformative observation."""
    for i, r in enumerate(rows):
        missing = [f for f in feature_names if r.get(f) is None]
        if missing:
            raise ValueError(f"row {i} is missing features {missing}; "
                             f"rows with undefined features are rejected")
    if stats is None:
        stats = {}
        for f in feature_names:
            vals = [r[f] for r in rows]
            mu = statistics.fmean(vals) if vals else 0.0
            sd = statistics.pstdev(vals) if len(vals) > 1 else 1.0
            stats[f] = (mu, sd if sd else 1.0)
    X = [[(r[f] - stats[f][0]) / stats[f][1] for f in feature_names]
         for r in rows]
    return X, stats


def fit_ridge_logistic(X, y, l2: float = 1.0, iters: int = 2000,
                       lr: float = 0.1) -> Dict[str, Any]:
    """Frozen hyperparameters (section 6a). stdlib only; no numpy or scipy."""
    n_f = len(X[0]) if X else 0
    w, b, n = [0.0] * n_f, 0.0, (len(X) or 1)
    for _ in range(iters):
        gw, gb = [0.0] * n_f, 0.0
        for xi, yi in zip(X, y):
            z = b + sum(wj * xj for wj, xj in zip(w, xi))
            p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))
            err = p - yi
            for j in range(n_f):
                gw[j] += err * xi[j]
            gb += err
        w = [wj - lr * (gw[j] / n + l2 * wj / n) for j, wj in enumerate(w)]
        b -= lr * gb / n

    def predict(x):
        z = b + sum(wj * xj for wj, xj in zip(w, x))
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    return {"w": w, "b": b, "predict": predict}


def auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Rank AUC, ties at 0.5. None when a class is absent -- never a defaulted
    0.5, which would look like a real chance result."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > q else 0.5 if p == q else 0.0)
               for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def bootstrap_auc_lower_bound(scores, labels, seed: int = BOOTSTRAP_SEED,
                              replicates: int = BOOTSTRAP_REPLICATES,
                              alpha: float = 0.05) -> Optional[float]:
    if auc(scores, labels) is None:
        return None
    rng = random.Random(seed)
    n = len(scores)
    vals = []
    for _ in range(replicates):
        idx = [rng.randrange(n) for _ in range(n)]
        a = auc([scores[i] for i in idx], [labels[i] for i in idx])
        if a is not None:
            vals.append(a)
    if not vals:
        return None
    vals.sort()
    return vals[int(alpha * len(vals))]


def evaluate_detector(discovery: Sequence[Dict[str, Any]],
                      validation: Sequence[Dict[str, Any]],
                      seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """Section 6's frozen bars. Fails CLOSED, and capacity is checked AFTER
    missing-feature rejection."""
    d = prepare_rows(discovery)
    v = prepare_rows(validation)
    v_pos, v_neg = v["y"].count(1), v["y"].count(0)
    base = {"n_misleading": v_pos, "n_stable_negative": v_neg,
            "rejected_missing_features": v["rejected_missing_features"],
            "dropped_ineligible": v["dropped_ineligible"],
            "auc": None, "auc_lower_bound": None}
    if v_pos < MIN_VALIDATION_MISLEADING or v_neg < MIN_VALIDATION_STABLE_NEGATIVE:
        return {**base, "verdict": "INSUFFICIENT_CLASSES",
                "reason": "validation split cannot support its own gate"}
    if d["y"].count(1) == 0 or d["y"].count(0) == 0:
        return {**base, "verdict": "INSUFFICIENT_DISCOVERY_CLASSES",
                "reason": "cannot fit with a single discovery class"}

    Xd, stats = standardize(d["features"])
    model = fit_ridge_logistic(Xd, d["y"])
    Xv, _ = standardize(v["features"], stats=stats)
    sv = [model["predict"](x) for x in Xv]

    a = auc(sv, v["y"])
    lb = bootstrap_auc_lower_bound(sv, v["y"], seed=seed)
    sd = sorted((model["predict"](x) for x in Xd), reverse=True)
    thr = sd[max(0, int(MAX_FLAG_RATE * len(sd)) - 1)] if sd else 1.0
    flagged = [(s, y) for s, y in zip(sv, v["y"]) if s >= thr]
    flag_rate = len(flagged) / len(sv) if sv else None
    precision = (sum(y for _s, y in flagged) / len(flagged)) if flagged else None

    passed = (a is not None and a >= AUC_BAR and lb is not None
              and lb >= AUC_LOWER_BOUND_BAR and flag_rate is not None
              and flag_rate <= MAX_FLAG_RATE and precision is not None
              and precision >= MIN_PRECISION)
    return {**base, "verdict": "PASS" if passed else "FAIL", "auc": a,
            "auc_lower_bound": lb, "flag_rate": flag_rate,
            "precision": precision, "threshold": thr}


def deployability(remaining_values: Sequence[int],
                  strata: Optional[Dict[str, Sequence[int]]] = None
                  ) -> Dict[str, Any]:
    """Section 6: remaining == 0 is non-actionable; a MEDIAN of zero fails the
    controller-deployability claim. Strata are REPORTED, never gated."""
    def summarize(vals: Sequence[int]) -> Dict[str, Any]:
        if not vals:
            return {"n": 0, "median_remaining": None,
                    "zero_budget_fraction": None, "quartiles": None}
        s = sorted(vals)
        return {"n": len(s), "median_remaining": statistics.median(s),
                "zero_budget_fraction": sum(1 for x in s if x == 0) / len(s),
                "quartiles": (statistics.quantiles(s, n=4, method="inclusive")
                              if len(s) >= 2 else None)}

    overall = summarize(remaining_values)
    med = overall["median_remaining"]
    verdict = ("NO_ROWS" if med is None
               else "NOT_DEPLOYABLE" if med == 0 else "DEPLOYABLE")
    return {**overall, "verdict": verdict,
            "by_stratum": {k: summarize(v) for k, v in (strata or {}).items()}}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_a.py -v -p no:cacheprovider`
Expected: PASS — 20 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_readout_a.py tests/test_atlas_readout_a.py
git commit -m "feat(atlas-s4): Read-out A string-label mapping, frozen bars, deployability"
```

---

### Task 4: Read-out B — four rungs, natural convergence, strata

**Files:**
- Create: `scripts/GPU/alphazero/atlas_readout_b.py`
- Test: `tests/test_atlas_readout_b.py`

**Interfaces:**
- Consumes: a **calibration row** — `{"legs": [...], "phase": str, "flat_policy": bool, "near_even": bool}`. A bare `list[LegResult]` cannot identify the strata §7 requires.
- Produces: `CalibrationRow` keys; `gate_triggers(legs, hi=1600)`; `closes_half(...)`; `convergent(legs, ref)`; `compound_narrowing(legs) -> Optional[bool]`; `calibrate_gate(rows, gate_name)`; `natural_convergence_report(rows)`; `by_stratum_summary(rows, gate_name)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_readout_b.py
import pytest

from scripts.GPU.alphazero.atlas_labelling import stable_reference
from scripts.GPU.alphazero.atlas_readout_b import (
    BASE_RATE_MARGIN, MIN_CONVERGENT_RATE, MIN_ELIGIBLE_TRIGGERS,
    by_stratum_summary, calibrate_gate, closes_half, compound_narrowing,
    convergent, gate_triggers, natural_convergence_report,
)
from scripts.GPU.alphazero.warm_prefix_replay import LegResult


def _leg(b, value, move, top_share=0.5, eff=12.0, rank=1, margin=0.20):
    return LegResult(nominal_B=b, inherited_I=10, effective=10 + b,
                     root_value=value, selected_move=move,
                     selected_move_prior_rank=rank, top_share=top_share,
                     top_two_margin=margin, effective_children=eff,
                     n_visited_children=20, visit_counts={move: 100})


def _legs(v=(0.9, 0.4, 0.05, 0.05), m=(7, 3, 3, 3), shares=(0.5, 0.6, 0.7, 0.7),
          effs=(20.0, 16.0, 12.0, 12.0), ranks=(1, 1, 1, 1)):
    return [_leg(b, v[i], m[i], shares[i], effs[i], ranks[i])
            for i, b in enumerate((400, 1600, 3200, 6400))]


def _row(legs=None, phase="late", flat=False, near_even=False):
    return {"legs": legs or _legs(), "phase": phase,
            "flat_policy": flat, "near_even": near_even}


def test_frozen_thresholds_are_pinned():
    assert MIN_ELIGIBLE_TRIGGERS == 10
    assert MIN_CONVERGENT_RATE == 0.75
    assert BASE_RATE_MARGIN == 0.15


def test_closes_half_needs_a_real_gap_and_half_closure():
    assert closes_half(1.0, 0.4, 0.0) is True
    assert closes_half(1.0, 0.8, 0.0) is False
    assert closes_half(0.5, 0.5, 0.5) is False       # no gap: vacuous


def test_convergent_requires_persistence_as_a_JOINT_condition():
    legs = _legs()
    assert convergent(legs, stable_reference(legs))["convergent"] is True
    broken = _legs(m=(7, 9, 3, 3))
    r = convergent(broken, stable_reference(broken))
    assert r["persistent"] is False and r["convergent"] is False


def test_dist_convergent_needs_the_SAME_metric_at_both_deep_rungs():
    ok = _legs(v=(0.06, 0.05, 0.05, 0.05), m=(3, 3, 3, 3),
               shares=(0.20, 0.60, 0.62, 0.62))
    assert convergent(ok, stable_reference(ok))["dist_convergent"] is True
    mixed = _legs(v=(0.06, 0.05, 0.05, 0.05), m=(3, 3, 3, 3),
                  shares=(0.20, 0.60, 0.90, 0.62))
    assert convergent(mixed, stable_reference(mixed))["dist_convergent"] is False


def test_gate_triggers_take_the_upper_rung_as_a_parameter():
    legs = _legs(shares=(0.90, 0.96, 0.97, 0.97))
    assert gate_triggers(legs, hi=1600)["new_collapse"] is True
    assert gate_triggers(legs, hi=6400)["new_collapse"] is True
    quiet = _legs(shares=(0.90, 0.91, 0.97, 0.97))
    assert gate_triggers(quiet, hi=1600)["new_collapse"] is False


def test_lower_prior_flip_uses_the_prior_RANK():
    legs = _legs(m=(3, 9, 9, 9), ranks=(1, 7, 7, 7))
    assert gate_triggers(legs, hi=1600)["lower_prior_flip"] is True


def test_compound_narrowing_is_an_AGGREGATE_not_a_per_row_boolean():
    """Mean effective-children reduction >= 0.50 AND mean top-share increase
    >= 0.15, over the cohort."""
    strong = [_row(_legs(shares=(0.50, 0.90, 0.90, 0.90),
                         effs=(20.0, 8.0, 8.0, 8.0))) for _ in range(4)]
    assert compound_narrowing(strong) is True          # 60% and +0.40
    # Narrowed, but nowhere near the aggregate thresholds -- a per-row
    # directional test would wrongly call this compound narrowing.
    slight = [_row(_legs(shares=(0.50, 0.52, 0.52, 0.52),
                         effs=(20.0, 19.0, 19.0, 19.0))) for _ in range(4)]
    assert compound_narrowing(slight) is False


def test_compound_narrowing_is_None_where_inapplicable():
    assert compound_narrowing([_row(_legs(shares=(None, None, None, None)))]) is None
    assert compound_narrowing([]) is None


def test_compound_narrowing_is_None_when_ANY_row_is_partially_missing():
    """Both means must describe the SAME cohort. A row contributing to one mean
    but not the other would make them summarize different row sets."""
    good = _row(_legs(shares=(0.50, 0.90, 0.90, 0.90),
                      effs=(20.0, 8.0, 8.0, 8.0)))
    partial = _row(_legs(shares=(0.50, 0.90, 0.90, 0.90),
                         effs=(None, None, None, None)))   # share only
    assert compound_narrowing([good] * 4) is True
    assert compound_narrowing([good] * 4 + [partial]) is None


def test_natural_convergence_report_covers_400_to_6400():
    rows = [_row() for _ in range(4)]
    r = natural_convergence_report(rows)
    assert set(r["trigger_rates"]) >= {"new_collapse", "top_share_increase"}
    # Reported as the reference distribution, NOT causal evidence that a
    # same-budget intervention is safe.
    assert r["is_causal_evidence"] is False
    assert r["transition"] == "400->6400"


def test_calibration_uses_the_ELIGIBLE_denominator():
    rows = [_row() for _ in range(12)] + [_row(_legs(m=(7, 3, 3, 9)))] * 5
    r = calibrate_gate(rows, "top_share_increase")
    assert r["eligible_triggers"] <= r["total_triggers"]
    assert r["eligible_trigger_fraction"] is not None


def test_needs_review_requires_all_three_conditions():
    """Each condition falsified individually -- accepting either verdict would
    prove nothing."""
    conv = [_row() for _ in range(12)]
    assert calibrate_gate(conv[:5], "top_share_increase")["verdict"] == "no finding"
    mixed = conv[:6] + [_row(_legs(m=(7, 9, 3, 3))) for _ in range(6)]
    rm = calibrate_gate(mixed, "top_share_increase")
    assert (rm["convergent_rate"] or 0) < MIN_CONVERGENT_RATE
    assert rm["verdict"] == "no finding"
    ra = calibrate_gate(conv, "top_share_increase")
    margin = ((ra["convergent_rate"] - ra["base_convergent_rate"])
              if ra["convergent_rate"] is not None else None)
    assert (ra["verdict"] == "needs review") == (
        ra["eligible_triggers"] >= MIN_ELIGIBLE_TRIGGERS
        and (ra["convergent_rate"] or 0) >= MIN_CONVERGENT_RATE
        and (margin or 0) >= BASE_RATE_MARGIN)
    assert "invalid" not in ra["verdict"]


def test_calibration_is_None_not_zero_with_no_eligible_triggers():
    rows = [_row(_legs(m=(7, 3, 3, 9)))] * 4
    r = calibrate_gate(rows, "new_collapse")
    assert r["convergent_rate"] is None and r["verdict"] == "no finding"


def test_stratum_summary_uses_the_ROW_schema_and_does_not_gate():
    rows = [_row(phase="late"), _row(phase="midgame", flat=True),
            _row(phase="late", near_even=True)]
    s = by_stratum_summary(rows, "top_share_increase")
    assert set(s) >= {"overall", "late", "flat_policy", "near_even"}
    # Section 7: no per-stratum acceptance gate.
    for k, v in s.items():
        if k != "overall":
            assert "verdict" not in v
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_b.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/atlas_readout_b.py
"""Atlas Read-out B -- design section 7, FROZEN.

Calibration, not a hypothesis: does an inherited collateral gate fire on changes
that move TOWARD the stable deeper reference?

Rows carry phase and flat/near-even facts, because a bare list of LegResults
cannot identify the strata section 7 requires.

The outcome is "needs review", never "invalid".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .atlas_labelling import stable_reference

COLLAPSE_TOP_SHARE = 0.95
VALUE_CONVERGENCE_TOL = 0.10
HALF = 0.5
MIN_ELIGIBLE_TRIGGERS = 10
MIN_CONVERGENT_RATE = 0.75
BASE_RATE_MARGIN = 0.15
GATE_NAMES = ("new_collapse", "lower_prior_flip",
              "effective_children_drop", "top_share_increase")


def _by_b(legs: Sequence[Any]) -> Dict[int, Any]:
    return {l.nominal_B: l for l in legs}


def gate_triggers(legs: Sequence[Any], hi: int = 1600) -> Dict[str, bool]:
    """Historical metrics on the 400 -> `hi` transition.

    `hi` is a parameter so the required 400 -> 6,400 natural-convergence report
    reuses this code rather than duplicating it.
    """
    d = _by_b(legs)
    a, b = d[400], d[hi]
    return {
        "new_collapse": (a.top_share is not None and b.top_share is not None
                         and a.top_share < COLLAPSE_TOP_SHARE
                         and b.top_share >= COLLAPSE_TOP_SHARE),
        "lower_prior_flip": (a.selected_move != b.selected_move
                             and a.selected_move_prior_rank is not None
                             and b.selected_move_prior_rank is not None
                             and b.selected_move_prior_rank
                             > a.selected_move_prior_rank),
        "effective_children_drop": (a.effective_children is not None
                                    and b.effective_children is not None
                                    and b.effective_children
                                    < a.effective_children),
        "top_share_increase": (a.top_share is not None and b.top_share is not None
                               and b.top_share > a.top_share),
    }


COMPOUND_EFF_CHILDREN_REDUCTION = 0.50
COMPOUND_TOP_SHARE_INCREASE = 0.15


def compound_narrowing(rows: Sequence[Dict[str, Any]], hi: int = 1600
                       ) -> Optional[bool]:
    """Section 7's compound condition -- an AGGREGATE over the cohort, not a
    per-row boolean (spec amendment).

    mean effective-children reduction >= 0.50 AND mean top-share increase
    >= 0.15. A per-row directional test would fire on any row that narrowed at
    all, however slightly, and would report compound narrowing where the
    historical gate saw none.

    None when the cohort has no defined aggregate -- inapplicable is not failing.
    """
    # COMPLETE CASE: both means must describe the SAME cohort. Accumulating them
    # independently lets a partially missing row contribute to one mean and not
    # the other, so the two would summarize different row sets.
    reductions, increases = [], []
    for row in rows:
        d = _by_b(row["legs"])
        a, b = d[400], d[hi]
        if (a.effective_children is None or b.effective_children is None
                or a.effective_children <= 0
                or a.top_share is None or b.top_share is None):
            return None            # any incomplete row makes the aggregate None
        reductions.append((a.effective_children - b.effective_children)
                          / a.effective_children)
        increases.append(b.top_share - a.top_share)
    if not reductions:
        return None
    mean_red = sum(reductions) / len(reductions)
    mean_inc = sum(increases) / len(increases)
    return bool(mean_red >= COMPOUND_EFF_CHILDREN_REDUCTION
                and mean_inc >= COMPOUND_TOP_SHARE_INCREASE)


def closes_half(m400: Optional[float], m1600: Optional[float],
                D: Optional[float]) -> bool:
    """|m400 - D| > 0 AND |m1600 - D| <= 0.5 * |m400 - D|.

    The `> 0` guard matters: with no gap, "closes half" is vacuous and must not
    fire rather than firing trivially.
    """
    if m400 is None or m1600 is None or D is None:
        return False
    gap = abs(m400 - D)
    if gap <= 0:
        return False
    return abs(m1600 - D) <= HALF * gap


def convergent(legs: Sequence[Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    """The FROZEN section 7 predicate. Persistence is a JOINT requirement."""
    d = _by_b(legs)
    deep = ref.get("stable_deep_move")
    move_conv = d[400].selected_move != deep and d[1600].selected_move == deep
    value_conv = (abs(d[1600].root_value - d[6400].root_value)
                  <= abs(d[400].root_value - d[6400].root_value)
                  - VALUE_CONVERGENCE_TOL)
    # SAME metric toward BOTH deep rungs. The disjunction is over METRICS, not
    # rungs: mixing one metric's 3,200 agreement with another's 6,400 is not
    # evidence.
    ts = (closes_half(d[400].top_share, d[1600].top_share, d[3200].top_share)
          and closes_half(d[400].top_share, d[1600].top_share, d[6400].top_share))
    ec = (closes_half(d[400].effective_children, d[1600].effective_children,
                      d[3200].effective_children)
          and closes_half(d[400].effective_children, d[1600].effective_children,
                          d[6400].effective_children))
    dist_conv = (d[1600].selected_move == deep) and (ts or ec)
    persistent = (d[1600].selected_move == d[3200].selected_move
                  == d[6400].selected_move)
    return {"move_convergent": move_conv, "value_convergent": value_conv,
            "dist_convergent": dist_conv, "persistent": persistent,
            "convergent": bool(persistent
                               and (move_conv or value_conv or dist_conv))}


def calibrate_gate(rows: Sequence[Dict[str, Any]], gate_name: str
                   ) -> Dict[str, Any]:
    """Section 7's frozen "needs review" rule, on the ELIGIBLE denominator."""
    total = eligible = confirmed = eligible_rows = base_conv = 0
    for row in rows:
        legs = row["legs"]
        ref = stable_reference(legs)
        fired = gate_triggers(legs)[gate_name]
        if fired:
            total += 1
        if not ref["stable"]:
            continue                      # unclassifiable: excluded, not counted
        eligible_rows += 1
        conv = convergent(legs, ref)["convergent"]
        base_conv += 1 if conv else 0
        if fired:
            eligible += 1
            confirmed += 1 if conv else 0

    rate = (confirmed / eligible) if eligible else None
    base_rate = (base_conv / eligible_rows) if eligible_rows else None
    needs_review = (eligible >= MIN_ELIGIBLE_TRIGGERS
                    and rate is not None and rate >= MIN_CONVERGENT_RATE
                    and base_rate is not None
                    and (rate - base_rate) >= BASE_RATE_MARGIN)
    return {"gate": gate_name, "total_triggers": total,
            "eligible_triggers": eligible,
            "eligible_trigger_fraction": ((eligible / total) if total else None),
            "confirmed_convergent": confirmed, "convergent_rate": rate,
            "base_convergent_rate": base_rate,
            # "needs review" means the gate structure must be reviewed and
            # frozen before it judges another prototype. It does NOT mean the
            # gate is invalid and does not authorize deleting or relaxing it.
            "verdict": "needs review" if needs_review else "no finding"}


def natural_convergence_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 7: the same metrics at 400 -> 6,400, to show the SCALE of natural
    deeper-search change.

    This is the natural-convergence reference distribution. It is explicitly
    NOT causal evidence that a same-budget intervention is safe.
    """
    n = len(rows)
    counts = {g: 0 for g in GATE_NAMES}
    for row in rows:
        t = gate_triggers(row["legs"], hi=6400)
        for g in GATE_NAMES:
            counts[g] += 1 if t[g] else 0
    return {"transition": "400->6400", "n_rows": n,
            "trigger_counts": counts,
            "trigger_rates": {g: (c / n if n else None) for g, c in counts.items()},
            "is_causal_evidence": False}


def by_stratum_summary(rows: Sequence[Dict[str, Any]], gate_name: str
                       ) -> Dict[str, Any]:
    """Overall plus late / flat-policy / near-even. Section 7 creates NO
    per-stratum acceptance gate, so the strata carry counts only."""
    def strip(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if k != "verdict"}

    out: Dict[str, Any] = {"overall": calibrate_gate(rows, gate_name)}
    for name, pred in (("late", lambda r: r["phase"] == "late"),
                       ("flat_policy", lambda r: r["flat_policy"]),
                       ("near_even", lambda r: r["near_even"])):
        subset = [r for r in rows if pred(r)]
        out[name] = strip(calibrate_gate(subset, gate_name)) if subset else {
            "n_rows": 0, "convergent_rate": None}
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_b.py -v -p no:cacheprovider`
Expected: PASS — 12 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_readout_b.py tests/test_atlas_readout_b.py
git commit -m "feat(atlas-s4): Read-out B four rungs, natural convergence and strata"
```

---

### Task 5: Read-out C — retention, aggregation, discovery-selected shape

**Files:**
- Create: `scripts/GPU/alphazero/atlas_readout_c.py`
- Test: `tests/test_atlas_readout_c.py`

**Interfaces:**
- Consumes: a **Read-out C row** — `{"reference_line": <Task 0 summary>, "snapshots": {...}, "label": str, "phase": str, "flat_policy": bool, "near_even": bool}`.
- Produces: `STRATA`; `static_retention(...)`; `intervention_from_snapshots(...)`; `classify_strata(row)`; `aggregate_shape(rows, shape)`; `select_shape(per_shape)`; `select_on_discovery_validate_on_selected(discovery, validation)`.

> **`K(n)` uses EFFECTIVE parent visits** (§6a). At the warm root that is
> `reference_line["root_effective_visits"]` — `I + N_actual` — **not** the nominal 320.
> Using the nominal would narrow the admitted set, understating retention and
> overstating intervention, in the same direction as the batch lag.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_readout_c.py
import pytest

from scripts.GPU.alphazero.atlas_readout_c import (
    MISLEADING_INTERVENTION_BAR, RETENTION_DEPTH1_BAR, RETENTION_ROOT_BAR,
    STABLE_INTERVENTION_CEILING, STRATA, aggregate_shape, classify_strata,
    classify_edge_strata, intervention_from_snapshots,
    select_on_discovery_validate_on_selected, select_shape, static_retention,
)

SHAPE = ("c4a05", 4.0, 0.5)


def _priors(n, best=0):
    return {i: (1.0 if i == best else 0.5 - i * 1e-4) for i in range(n)}


def _snap(ft=100, ft_out=15, lagged=12, elig=200):
    return {"at_boundary": {"by_shape": {"c4a05": {"overall": {
        "eligible_events": elig, "outside_events": 30,
        "first_touch_events": ft, "first_touch_outside_events": ft_out,
        "lagged_first_touch_outside_events": lagged,
        "excluded_prior_mass": 0.4}}}}}


def _ref(root_n=326 + 137, reply_n=90, root_move=0, reply_move=1):
    return {"stable_deep_move": root_move, "root_priors": _priors(500),
            "root_effective_visits": root_n,
            "reply": {"move": reply_move, "priors": _priors(400),
                      "effective_visits": reply_n},
            "two_ply": None}


def _row(label="misleading", phase="late", flat=False, near_even=False,
         ref=None, snaps=None):
    return {"reference_line": ref or _ref(), "snapshots": snaps or _snap(),
            "label": label, "phase": phase, "flat_policy": flat,
            "near_even": near_even}


def test_frozen_bars_and_strata_are_pinned():
    assert RETENTION_ROOT_BAR == 0.95 and RETENTION_DEPTH1_BAR == 0.90
    assert MISLEADING_INTERVENTION_BAR == 0.50
    assert STABLE_INTERVENTION_CEILING == 0.25
    assert set(STRATA) == {"late", "near_even", "root_flat",
                           "locally_flat_depth1", "locally_flat_depth2"}


def test_static_retention_uses_EFFECTIVE_parent_visits():
    """K(n) keys on completed visits, which at a warm root include I."""
    wide = static_retention(_priors(500), [80], n_at_selection=463, shape=SHAPE)
    narrow = static_retention(_priors(500), [80], n_at_selection=320, shape=SHAPE)
    assert wide["k"] > narrow["k"]
    assert wide["retained"] == 1 and narrow["retained"] == 0


def test_static_retention_of_nothing_is_None():
    assert static_retention(_priors(10), [], 400, SHAPE)["rate"] is None


def test_intervention_requires_the_PRODUCED_lagged_bound():
    r = intervention_from_snapshots(_snap(ft=100, ft_out=12, lagged=8), "c4a05")
    assert r["meaningfully_affected"] is None      # None, not False
    assert r["verdict"] == "INCONCLUSIVE"


def test_intervention_passes_when_both_bounds_clear():
    r = intervention_from_snapshots(_snap(ft=100, ft_out=15, lagged=12), "c4a05")
    assert r["meaningfully_affected"] is True and r["verdict"] == "OK"


def test_classify_strata_reads_the_row_not_a_bare_leg_list():
    s = classify_strata(_row(phase="late", flat=True, near_even=True))
    assert {"late", "root_flat", "near_even"} <= s


def test_local_flat_strata_are_EDGE_level_not_row_level():
    """A row can hold both flat and non-flat reference parents; pooling them
    would hide the contrast the stratum exists to expose."""
    flat = {"depth": 1, "parent_priors": {i: 1.0 / 500 for i in range(500)}}
    sharp = {"depth": 1, "parent_priors": {0: 0.9, 1: 0.05, 2: 0.05}}
    assert "locally_flat_depth1" in classify_edge_strata(flat)
    assert classify_edge_strata(sharp) == set()


def test_aggregate_excludes_INCONCLUSIVE_rows_from_the_denominator():
    """Folding them in as either outcome would invent a measurement."""
    rows = [_row(snaps=_snap(ft=100, ft_out=15, lagged=12)),        # OK, affected
            _row(snaps=_snap(ft=100, ft_out=12, lagged=8))]         # inconclusive
    a = aggregate_shape(rows, SHAPE)
    assert a["misleading_denominator"] == 1
    assert a["inconclusive"] == 1


def test_aggregate_rate_is_None_when_the_denominator_empties():
    rows = [_row(snaps=_snap(ft=100, ft_out=12, lagged=8))]
    a = aggregate_shape(rows, SHAPE)
    assert a["misleading_intervention"] is None


def test_a_shape_with_a_None_rate_cannot_pass():
    per = {"c4a05": {"root_retention": 0.99, "depth1_retention": 0.95,
                     "misleading_intervention": None, "stable_intervention": 0.10,
                     "descendant_retention": 0.90}}
    assert select_shape(per)["selected"] is None


def test_shape_selection_is_lexicographic():
    a = {"root_retention": 0.99, "depth1_retention": 0.95,
         "misleading_intervention": 0.60, "stable_intervention": 0.20,
         "descendant_retention": 0.80}
    b = dict(a, misleading_intervention=0.55, stable_intervention=0.10,
             descendant_retention=0.99)
    assert select_shape({"c4a05": a, "c13a03": b})["selected"] == "c4a05"


def test_a_retention_floor_excludes_a_shape_however_good_otherwise():
    a = {"root_retention": 0.90, "depth1_retention": 0.95,
         "misleading_intervention": 0.99, "stable_intervention": 0.01,
         "descendant_retention": 0.99}
    b = {"root_retention": 0.96, "depth1_retention": 0.91,
         "misleading_intervention": 0.51, "stable_intervention": 0.24,
         "descendant_retention": 0.70}
    assert select_shape({"c4a05": a, "c13a03": b})["selected"] == "c13a03"


def test_no_shape_passing_is_a_named_failure():
    bad = {"root_retention": 0.10, "depth1_retention": 0.10,
           "misleading_intervention": 0.99, "stable_intervention": 0.99,
           "descendant_retention": 0.10}
    r = select_shape({"c4a05": bad, "c13a03": bad})
    assert r["selected"] is None and r["verdict"] == "NO_SHAPE_PASSES"


def test_ties_break_on_descendant_retention():
    a = {"root_retention": 0.99, "depth1_retention": 0.95,
         "misleading_intervention": 0.60, "stable_intervention": 0.20,
         "descendant_retention": 0.70}
    b = dict(a, descendant_retention=0.90)
    assert select_shape({"c4a05": a, "c13a03": b})["selected"] == "c13a03"


def test_selection_happens_on_discovery_and_only_that_shape_is_validated():
    disc = [_row() for _ in range(4)]
    val = [_row() for _ in range(4)]
    r = select_on_discovery_validate_on_selected(disc, val)
    assert r["selected_on"] == "discovery"
    assert set(r["validated"]) <= {r["selected"]}      # never both shapes
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_c.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/atlas_readout_c.py
"""Atlas Read-out C -- design section 8 and amendment 6a, FROZEN.

Counterfactual COVERAGE analysis. It cannot prove progressive widening would
improve search, because applying widening changes the later tree.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Set, Tuple

from .selection_tracer import (
    MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE, WIDENING_SHAPES, k_of_n,
)

RETENTION_ROOT_BAR = 0.95
RETENTION_DEPTH1_BAR = 0.90
MISLEADING_INTERVENTION_BAR = 0.50
STABLE_INTERVENTION_CEILING = 0.25
FLAT_ENTROPY_BAR = 0.90
FLAT_TOP_PRIOR_BAR = 0.025

STRATA = ("late", "near_even", "root_flat",
          "locally_flat_depth1", "locally_flat_depth2")


def static_retention(root_priors: Dict[int, float],
                     required_moves: Sequence[int], n_at_selection: int,
                     shape: Tuple[str, float, float]) -> Dict[str, Any]:
    """Would the moves stable deeper search requires have been admitted?

    `n_at_selection` is the parent's EFFECTIVE completed visit count -- at a warm
    root that is I + N_actual, never the nominal 320 (amendment 6a). Retention is
    evaluated under K(n), the narrower conservative set, so a pass is safe.
    """
    _name, c, alpha = shape
    n_legal = len(root_priors)
    k = k_of_n(n_at_selection, c, alpha, n_legal)
    order = sorted(root_priors.items(), key=lambda kv: (-kv[1], kv[0]))
    rank = {mv: i + 1 for i, (mv, _p) in enumerate(order)}
    if not required_moves:
        return {"retained": 0, "required": 0, "rate": None, "k": k}
    retained = sum(1 for mv in required_moves if rank.get(mv, n_legal + 1) <= k)
    return {"retained": retained, "required": len(required_moves),
            "rate": retained / len(required_moves), "k": k}


def intervention_from_snapshots(snapshots: Dict[str, Any], shape_key: str
                                ) -> Dict[str, Any]:
    """Meaningful intervention with the DIRECTIONAL lag bound.

    The lag is conservative for retention and ANTI-conservative for
    intervention, so the threshold must also pass under K(n+14) -- a counter the
    tracer PRODUCES, never a caller-supplied number. Passing only under K(n) is
    INCONCLUSIVE, not a pass.
    """
    cell = snapshots["at_boundary"]["by_shape"][shape_key]["overall"]
    ft = cell["first_touch_events"]
    if not ft:
        return {"first_touch_outside_rate": None, "lagged_rate": None,
                "meaningfully_affected": None, "verdict": "NO_EVENTS"}
    rate = cell["first_touch_outside_events"] / ft
    lagged = cell["lagged_first_touch_outside_events"] / ft
    ok = rate >= MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE
    lagged_ok = lagged >= MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE
    if ok and lagged_ok:
        verdict, affected = "OK", True
    elif ok:
        # None, never False: undecided is not a measured negative.
        verdict, affected = "INCONCLUSIVE", None
    else:
        verdict, affected = "OK", False
    return {"first_touch_outside_rate": rate, "lagged_rate": lagged,
            "meaningfully_affected": affected, "verdict": verdict}


def classify_strata(row: Dict[str, Any]) -> Set[str]:
    """Frozen strata from the ROW. Flat-policy status is recomputed LOCALLY
    along the reference line, not inherited from the root."""
    s: Set[str] = set()
    if row.get("phase") == "late":
        s.add("late")
    if row.get("near_even"):
        s.add("near_even")
    if row.get("flat_policy"):
        s.add("root_flat")
    return s


def classify_edge_strata(edge: Dict[str, Any]) -> Set[str]:
    """Locally-flat strata are EDGE-LEVEL, not row-level.

    Under the union-of-two-deep-lines rule a single row can contain both flat
    and non-flat reference parents. A row-level "any parent is flat" flag would
    pool their retention into one number and hide exactly the contrast the
    stratum exists to expose. Each deduplicated required edge is classified
    using ITS OWN parent priors.
    """
    s: Set[str] = set()
    priors = edge.get("parent_priors")
    if priors and _is_flat(priors):
        s.add("locally_flat_depth1" if edge.get("depth") == 1
              else "locally_flat_depth2")
    return s


def _is_flat(priors: Dict[int, float]) -> bool:
    """The FROZEN flat-policy definition: normalized entropy >= 0.90 AND top
    prior <= 0.025. Checking only the top prior misclassifies a concentrated
    low-top distribution as flat -- both halves are required."""
    vals = [p for p in priors.values() if p > 0]
    if len(vals) < 2:
        return False
    s = sum(vals)
    norm = [v / s for v in vals]
    entropy = (-sum(q * math.log(q) for q in norm)) / math.log(len(priors))
    return entropy >= FLAT_ENTROPY_BAR and max(norm) <= FLAT_TOP_PRIOR_BAR


def aggregate_shape(rows: Sequence[Dict[str, Any]],
                    shape: Tuple[str, float, float]) -> Dict[str, Any]:
    """Fold per-row three-valued results into the four rates section 8 gates.

    INCONCLUSIVE rows are excluded from the intervention denominator and counted
    separately -- folding them in as either outcome would invent a measurement.
    If a denominator empties, the rate is None and the shape CANNOT pass.
    """
    name = shape[0]
    root_ret = {"retained": 0, "required": 0}
    d1_ret = {"retained": 0, "required": 0}
    desc_ret = {"retained": 0, "required": 0}
    mis_num = mis_den = st_num = st_den = inconclusive = 0

    for row in rows:
        ref = row["reference_line"]
        r = static_retention(ref["root_priors"], [ref["stable_deep_move"]],
                             ref["root_effective_visits"], shape)
        root_ret["retained"] += r["retained"]; root_ret["required"] += r["required"]
        reply = ref.get("reply")
        if reply and reply.get("move") is not None:
            rr = static_retention(reply["priors"], [reply["move"]],
                                  reply["effective_visits"], shape)
            d1_ret["retained"] += rr["retained"]; d1_ret["required"] += rr["required"]
        two = ref.get("two_ply")
        if two and two.get("move") is not None:
            tr = static_retention(two["priors"], [two["move"]],
                                  two["effective_visits"], shape)
            desc_ret["retained"] += tr["retained"]; desc_ret["required"] += tr["required"]

        iv = intervention_from_snapshots(row["snapshots"], name)
        if iv["meaningfully_affected"] is None:
            inconclusive += 1
            continue
        if row["label"] == "misleading":
            mis_den += 1; mis_num += 1 if iv["meaningfully_affected"] else 0
        elif row["label"] == "stable_negative":
            st_den += 1; st_num += 1 if iv["meaningfully_affected"] else 0

    def rate(n, d):
        return (n / d) if d else None

    return {
        "shape": name,
        "root_retention": rate(root_ret["retained"], root_ret["required"]),
        "depth1_retention": rate(d1_ret["retained"], d1_ret["required"]),
        "descendant_retention": rate(desc_ret["retained"], desc_ret["required"]),
        "misleading_intervention": rate(mis_num, mis_den),
        "stable_intervention": rate(st_num, st_den),
        "misleading_denominator": mis_den, "stable_denominator": st_den,
        "inconclusive": inconclusive,
    }


def select_shape(per_shape: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Frozen LEXICOGRAPHIC order: retention floors -> stable-intervention
    ceiling -> higher misleading intervention -> tie on descendant retention.

    A None rate cannot pass: an undefined denominator is not a satisfied bar.
    """
    def passes(v):
        return (v.get("root_retention") is not None
                and v["root_retention"] >= RETENTION_ROOT_BAR
                and v.get("depth1_retention") is not None
                and v["depth1_retention"] >= RETENTION_DEPTH1_BAR
                and v.get("stable_intervention") is not None
                and v["stable_intervention"] <= STABLE_INTERVENTION_CEILING
                and v.get("misleading_intervention") is not None)

    survivors = {k: v for k, v in per_shape.items() if passes(v)}
    if not survivors:
        return {"selected": None, "verdict": "NO_SHAPE_PASSES",
                "considered": sorted(per_shape)}
    best = max(survivors, key=lambda k: (
        survivors[k]["misleading_intervention"],
        survivors[k].get("descendant_retention") or 0.0))
    return {"selected": best, "verdict": "OK", "survivors": sorted(survivors),
            "considered": sorted(per_shape)}


def select_on_discovery_validate_on_selected(
        discovery: Sequence[Dict[str, Any]],
        validation: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose on DISCOVERY, then evaluate ONLY the selected shape on VALIDATION.

    Evaluating both on validation would let a shape be chosen for looking good
    on the split that judges it.
    """
    disc = {s[0]: aggregate_shape(discovery, s) for s in WIDENING_SHAPES}
    chosen = select_shape(disc)
    if chosen["selected"] is None:
        return {**chosen, "selected_on": "discovery", "validated": {}}
    shape = next(s for s in WIDENING_SHAPES if s[0] == chosen["selected"])
    return {**chosen, "selected_on": "discovery",
            "discovery": disc,
            "validated": {chosen["selected"]: aggregate_shape(validation, shape)}}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_c.py -v -p no:cacheprovider`
Expected: PASS — 14 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_readout_c.py tests/test_atlas_readout_c.py
git commit -m "feat(atlas-s4): Read-out C aggregation, strata and discovery-selected shape"
```

---

### Task 6: Artifact schema, provenance and `_jsonable`

**Files:**
- Create: `scripts/GPU/alphazero/atlas_artifact.py`
- Test: `tests/test_atlas_artifact.py`

**Interfaces:**
- Produces: `ROW_SCHEMA_VERSION`; `build_row(...)` carrying **both** `features_at_boundary` and `features_at_400`; `validate_provenance(run)`; `emit(run)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_artifact.py
import json

import pytest

from scripts.GPU.alphazero.atlas_artifact import (
    ROW_SCHEMA_VERSION, build_row, emit, validate_provenance,
)


def _kw(**over):
    base = dict(
        game_idx=3, replay_seed=20400003, target_ply=95, phase="late",
        side="black", split="validation", inherited_I=137,
        reset_count=1, reset_rate=0.02, last_reset_ply=44,
        boundary={"N_actual": 326, "overshoot": 6, "remaining": 74,
                  "flush_type": "full"},
        legs=[{"nominal_B": 400}], label="misleading",
        features_at_boundary={"one_visit_backup_share": 0.4},
        features_at_400={"one_visit_backup_share": 0.3},
        reference_line={"stable_deep_move": 7},
        tracer_snapshots={"at_boundary": {}, "at_400": {}},
        flat_policy=False, near_even=True)
    base.update(over)
    return base


def test_row_carries_BOTH_feature_captures():
    """Section 6a: B=400 supplies the required 400-tree diagnostic contrast, so
    it must survive into the artifact, not just the boundary capture."""
    r = build_row(**_kw())
    assert r["features_at_boundary"]["one_visit_backup_share"] == 0.4
    assert r["features_at_400"]["one_visit_backup_share"] == 0.3


def test_row_carries_resets_remaining_strata_and_the_schema_version():
    r = build_row(**_kw())
    assert r["schema_version"] == ROW_SCHEMA_VERSION
    assert r["reset_count"] == 1 and r["reset_rate"] == 0.02
    assert r["last_reset_ply"] == 44 and r["boundary"]["remaining"] == 74
    assert r["near_even"] is True and r["flat_policy"] is False


def test_undefined_values_stay_None_through_emission():
    r = build_row(**_kw(reset_rate=None, last_reset_ply=None, boundary=None,
                        features_at_400=None))
    back = json.loads(emit({"rows": [r], "provenance": {}}))["rows"][0]
    assert back["reset_rate"] is None and back["last_reset_ply"] is None
    assert back["boundary"] is None and back["features_at_400"] is None


def test_a_row_missing_the_boundary_is_flagged_not_defaulted():
    assert build_row(**_kw(boundary=None))["boundary_missing"] is True


def test_emission_goes_through_jsonable():
    run = {"rows": [], "provenance": {},
           "cells": {("discovery", "late", "red"): 12}}
    assert json.loads(emit(run))["cells"] == {"discovery|late|red": 12}


def test_emission_REJECTS_an_unserializable_payload():
    """No default=str: it would stringify a schema defect into a
    plausible-looking value instead of failing."""
    with pytest.raises(TypeError):
        emit({"rows": [{"bad": object()}], "provenance": {}})


def test_provenance_fails_closed_on_a_dirty_tree():
    r = validate_provenance({"git_head": "a" * 40, "worktree_clean": False,
                             "checkpoint_sha1": "0" * 40})
    assert r["verdict"] == "PROVENANCE_FAILURE" and "worktree_clean" in r["problems"]


def test_provenance_requires_a_checkpoint_digest():
    r = validate_provenance({"git_head": "a" * 40, "worktree_clean": True,
                             "checkpoint_sha1": ""})
    assert "checkpoint_sha1" in r["problems"]


def test_valid_provenance_passes():
    r = validate_provenance({"git_head": "a" * 40, "worktree_clean": True,
                             "checkpoint_sha1": "0" * 40})
    assert r["verdict"] == "OK" and r["problems"] == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_artifact.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/atlas_artifact.py
"""Atlas artifact schema, provenance validation and emission.

Every undefined value stays None through emission. A missing boundary is
FLAGGED, never defaulted -- a zero-filled record is indistinguishable from a
real one.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .build_atlas_corpus import _jsonable

ROW_SCHEMA_VERSION = 2


def build_row(*, game_idx: int, replay_seed: int, target_ply: int, phase: str,
              side: str, split: str, inherited_I: int, reset_count: int,
              reset_rate: Optional[float], last_reset_ply: Optional[int],
              boundary: Optional[Dict[str, Any]], legs: List[Dict[str, Any]],
              label: str, features_at_boundary: Optional[Dict[str, Any]],
              features_at_400: Optional[Dict[str, Any]],
              reference_line: Optional[Dict[str, Any]],
              tracer_snapshots: Dict[str, Any], flat_policy: bool,
              near_even: bool) -> Dict[str, Any]:
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "game_idx": game_idx, "replay_seed": replay_seed,
        "target_ply": target_ply, "phase": phase, "side": side, "split": split,
        "inherited_I": inherited_I,
        # Section 2b: reset statistics are explicit, and every row is kept.
        "reset_count": reset_count, "reset_rate": reset_rate,
        "last_reset_ply": last_reset_ply,
        "boundary": boundary, "boundary_missing": boundary is None,
        "legs": legs, "label": label,
        # BOTH captures: B=400 supplies section 6's 400-tree diagnostic contrast.
        "features_at_boundary": features_at_boundary,
        "features_at_400": features_at_400,
        "reference_line": reference_line,
        "tracer_snapshots": tracer_snapshots,
        # Strata facts, so Read-outs B and C need no second source.
        "flat_policy": flat_policy, "near_even": near_even,
    }


def validate_provenance(prov: Dict[str, Any]) -> Dict[str, Any]:
    """Fails CLOSED. A dirty tree or unidentifiable checkpoint means the run is
    not reconstructible, whatever its numbers say."""
    problems = []
    if prov.get("worktree_clean") is not True:
        problems.append("worktree_clean")
    sha1 = prov.get("checkpoint_sha1")
    if not sha1 or len(sha1) != 40:
        problems.append("checkpoint_sha1")
    head = prov.get("git_head")
    if not head or len(head) != 40:
        problems.append("git_head")
    return {"verdict": "PROVENANCE_FAILURE" if problems else "OK",
            "problems": problems}


def emit(run: Dict[str, Any]) -> str:
    """Serialize through _jsonable. NO default=str -- it would stringify a
    schema defect into a plausible-looking value instead of failing."""
    return json.dumps(_jsonable(run), indent=2, sort_keys=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_artifact.py -v -p no:cacheprovider`
Expected: PASS — 9 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_artifact.py tests/test_atlas_artifact.py
git commit -m "feat(atlas-s4): artifact carrying both captures, fail-closed provenance"
```

---

### Task 7: The real chain — Stage 3 output through all three read-outs

**Files:**
- Create: `tests/test_atlas_readout_chain.py`

**Interfaces:**
- Consumes: everything. Produces no new API.

> One shared CPU fixture at the **frozen increments**. Tiny ones give nominal budgets
> 8/16/24/32, which labelling and gate calibration reject outright since both index
> 400/1,600/3,200/6,400 — the "real chain" would raise before its first assertion.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_readout_chain.py
"""Real Stage 3 output -> real Read-outs A, B and C. No surrogates."""
import json
import random

import pytest

from scripts.GPU.alphazero.atlas_artifact import build_row, emit
from scripts.GPU.alphazero.atlas_labelling import class_counts, classify_row
from scripts.GPU.alphazero.atlas_readout_a import (
    collect_features, deployability, evaluate_detector,
)
from scripts.GPU.alphazero.atlas_readout_b import (
    by_stratum_summary, calibrate_gate, natural_convergence_report,
)
from scripts.GPU.alphazero.atlas_readout_c import (
    aggregate_shape, classify_strata, intervention_from_snapshots,
)
from scripts.GPU.alphazero.corpus_geometry import GameMeta
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.selection_tracer import SelectionTracer
from scripts.GPU.alphazero.warm_prefix_replay import (
    BatchSafeBoundaryObserver, replay_prefix, run_additive_ladder,
)

from tests.eval_fakes import FakeEvaluator

BASE = 20400000
SIZE = 6
SHAPE = ("c4a05", 4.0, 0.5)


def _history(n, size=SIZE):
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    s = TwixtState(active_size=size, to_move="red")
    out = []
    for _ in range(n):
        lm = s.legal_moves()
        if not lm:
            break
        out.append(lm[0])
        s = s.apply_move(lm[0])
    return out


@pytest.fixture(scope="module")
def real_row():
    """ONE shared fixture at the FROZEN increments -- 6,400 FakeEvaluator
    simulations at active_size=6, CPU-only."""
    hist = _history(4)
    meta = GameMeta(game_id=0, seed=BASE, n_moves=len(hist), start_player="red")
    m = MCTS(FakeEvaluator(value=0.0),
             MCTSConfig(n_simulations=1, eval_batch_size=14,
                        stall_flush_sims=48, pending_virtual_visits=8),
             random.Random(BASE))
    pre = replay_prefix(m, meta, hist, target_ply=2, active_size=SIZE)
    tracer = SelectionTracer()
    m._selection_observer = tracer
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I, tracer=tracer)
    legs, snaps = run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                                      boundary_observer=obs,
                                      target_tracer=tracer)   # frozen defaults
    return {"meta": meta, "pre": pre, "legs": legs, "snaps": snaps, "obs": obs}


def test_the_fixture_uses_the_frozen_rungs(real_row):
    assert [l.nominal_B for l in real_row["legs"]] == [400, 1600, 3200, 6400]


def test_real_legs_classify_and_count(real_row):
    label = classify_row(real_row["legs"])
    assert label in {"misleading", "stable_negative", "ambiguous",
                     "no_stable_reference"}
    assert set(class_counts([real_row["legs"]])) >= {"misleading",
                                                     "stable_negative"}


def test_readout_A_features_come_from_the_frozen_captures(real_row):
    caps, obs = real_row["snaps"]["captures"], real_row["obs"]
    f = collect_features(caps["at_start"], caps["at_boundary"],
                         obs.record.N_actual)
    assert set(f) == {"one_visit_backup_share", "depth3plus_backup_fraction",
                      "leader_visit_margin", "root_policy_entropy",
                      "leader_breadth"}
    f400 = collect_features(caps["at_start"], caps["at_400"], 400)
    assert f400 is not None            # the 400-tree diagnostic contrast


def test_readout_A_detector_runs_end_to_end_on_real_features(real_row):
    caps, obs = real_row["snaps"]["captures"], real_row["obs"]
    f = collect_features(caps["at_start"], caps["at_boundary"],
                         obs.record.N_actual)
    rows = ([{"label": "misleading", "features": f} for _ in range(20)]
            + [{"label": "stable_negative", "features": f} for _ in range(25)])
    r = evaluate_detector(discovery=rows, validation=rows)
    # Identical features cannot separate; the point is that the REAL path runs
    # and reports a verdict rather than raising.
    assert r["verdict"] in {"PASS", "FAIL", "INSUFFICIENT_CLASSES",
                            "INSUFFICIENT_DISCOVERY_CLASSES"}
    d = deployability([real_row["obs"].record.remaining])
    assert d["verdict"] in {"DEPLOYABLE", "NOT_DEPLOYABLE"}


def test_readout_B_runs_on_real_rows(real_row):
    row = {"legs": real_row["legs"], "phase": "opening",
           "flat_policy": False, "near_even": True}
    assert calibrate_gate([row], "top_share_increase")["verdict"] in {
        "needs review", "no finding"}
    nc = natural_convergence_report([row])
    assert nc["transition"] == "400->6400" and nc["is_causal_evidence"] is False
    assert "overall" in by_stratum_summary([row], "top_share_increase")


def test_readout_C_runs_on_the_real_reference_line(real_row):
    ref = real_row["snaps"]["reference_lines"]
    row = {"reference_line": ref, "snapshots": real_row["snaps"],
           "label": "misleading", "phase": "opening",
           "flat_policy": False, "near_even": True}
    assert isinstance(classify_strata(row), set)
    iv = intervention_from_snapshots(real_row["snaps"], "c4a05")
    assert iv["verdict"] in {"OK", "INCONCLUSIVE", "NO_EVENTS"}
    agg = aggregate_shape([row], SHAPE)
    assert set(agg) >= {"root_retention", "misleading_intervention",
                        "inconclusive"}


def test_a_real_row_survives_the_artifact_boundary(real_row):
    pre, legs, snaps, obs = (real_row["pre"], real_row["legs"],
                             real_row["snaps"], real_row["obs"])
    caps = snaps["captures"]
    row = build_row(
        game_idx=real_row["meta"].game_id, replay_seed=real_row["meta"].seed,
        target_ply=2, phase="opening", side="red", split="discovery",
        inherited_I=pre.inherited_I, reset_count=pre.reset_count,
        reset_rate=pre.reset_rate, last_reset_ply=pre.last_reset_ply,
        boundary=(vars(obs.record) if obs.record else None),
        legs=[vars(l) for l in legs], label=classify_row(legs),
        features_at_boundary=collect_features(caps["at_start"],
                                              caps["at_boundary"],
                                              obs.record.N_actual),
        features_at_400=collect_features(caps["at_start"], caps["at_400"], 400),
        reference_line=snaps["reference_lines"], tracer_snapshots=snaps,
        flat_policy=False, near_even=True)
    back = json.loads(emit({"rows": [row], "provenance": {}}))["rows"][0]
    assert back["inherited_I"] == pre.inherited_I
    assert len(back["legs"]) == 4
    assert back["features_at_boundary"] is not None
    assert back["features_at_400"] is not None
```

- [ ] **Step 2: Run, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_chain.py -v -p no:cacheprovider`
Expected: PASS — 7 passed.

```bash
.venv/bin/python -m pytest -p no:cacheprovider -q > /tmp/s4.out 2>&1; echo "REAL_EXIT=$?" >> /tmp/s4.out; tail -3 /tmp/s4.out
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_atlas_readout_chain.py
git commit -m "test(atlas-s4): real Stage 3 output through all three read-outs"
```

---

## Stage 4 completion criteria

- [ ] **Task 0 producer closure lands first.** Backup accounting uses §6a's two-point `D3` measurement, not selection events; features are frozen at the boundary and at `B=400` while those states exist; the tracer counts `K(n+14)` online; the backup invariant is asserted and a violation fails the row.
- [ ] §5 labelling exact: stable reference needs all three conditions; misleading is an OR; stable-negative an AND; ambiguous kept and counted; value and move components reported separately.
- [ ] §3 sizing matches the frozen formula and **fails closed** on a zero class frequency or a requirement above 400; the final capacity gate needs ≥20 / ≥25.
- [ ] Read-out A collects **exactly five** frozen features; standardization is learned on **discovery only**; AUC / bootstrap-lower-bound / flag-rate / precision bars all enforced; `INSUFFICIENT_CLASSES` fails closed.
- [ ] Deployability: `remaining == 0` non-actionable, **median zero fails**, strata **reported not gated**, empty set yields `None`.
- [ ] Read-out B computes metrics at **all four rungs**; `closes_half` guards a zero gap and checks **both** deep rungs for the **same** metric; persistence is joint; the denominator is `eligible_triggers` with the **base-rate margin**; the verdict is *needs review*, never *invalid*.
- [ ] Read-out C: retention under `K(n)`; intervention must **also** pass under `K(n+14)`, otherwise **inconclusive**; lexicographic selection with a named `NO_SHAPE_PASSES`.
- [ ] Artifact carries resets, `remaining`, `boundary_missing`, and `None` for every undefined value through emission; provenance fails closed; `_jsonable` at the boundary.
- [ ] Real Stage 3 ladder output drives the real read-outs.
- [ ] **78 new tests** (labelling 10, producer-closure 6, read-out A 20, read-out B 12, read-out C 14, artifact 9, chain 7). Full suite **2513 passed**, exit code read from the process.

## Out of scope

No reservoir generation, no checkpoint loading, no MLX execution, no measurement run.
The three distribution gaps — real-scale throughput, the `remaining` distribution, and
the inheritance-reset rate — remain **operator/pilot measurements**. Stage 5 is planned
only after these interfaces exist and qualify.
