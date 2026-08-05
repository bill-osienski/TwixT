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
    def __init__(self, visits, kids=None):
        self.visit_count = visits
        self.children = kids or {}


def test_D3_counts_nodes_at_depth_exactly_three():
    """Every backup reaching depth >=3 passes through exactly ONE depth-3 node,
    so summing visits there counts each such backup once."""
    root = _N(10, {0: _N(6, {0: _N(4, {0: _N(3), 1: _N(1)})}),
                   1: _N(4, {0: _N(2, {0: _N(2)})})})
    acc = capture_tree_state(root)
    assert acc["D3"] == 3 + 1 + 2          # depth-3 nodes only
    assert acc["n_visited_children"] == 2
    assert acc["one_visit_children"] == 0


def test_one_visit_children_is_counted_on_the_root_only():
    root = _N(5, {0: _N(3), 1: _N(1), 2: _N(1), 3: _N(0)})
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

### Task 2: Read-out A — boundary feature collector

**Files:**
- Create: `scripts/GPU/alphazero/atlas_readout_a.py`
- Test: `tests/test_atlas_readout_a.py`

**Interfaces:**
- Consumes: a root node (or a synthetic stand-in exposing `priors`/`children`) and a Stage 3 tracer snapshot.
- Produces: `FEATURE_NAMES`; `collect_boundary_features(capture_start, capture_boundary, n_actual, root_priors, leader_breadth) -> dict`.

> Consumes Task 0's **frozen captures**, never a live root: by the time Stage 4 runs, the root has advanced to 6,400. The two backup features use §6a's accounting, not selection events.

The five frozen §6 features, and **only** these five. Q dispersion, residual summaries
and terminating-backup concentration may be reported descriptively elsewhere but must
not enter the detector — §6 forbids an unrestricted feature search.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_readout_a.py
import math

import pytest

from scripts.GPU.alphazero.atlas_readout_a import (
    FEATURE_NAMES, collect_boundary_features,
)


class _Child:
    def __init__(self, visits, n_grandchildren=0):
        self.visit_count = visits
        self.children = {i: _Child(0) for i in range(n_grandchildren)}


class _Root:
    def __init__(self, priors, child_visits, leader_breadth=0):
        self.priors = priors
        self.children = {mv: _Child(v) for mv, v in child_visits.items()}
        if self.children:
            top = max(self.children, key=lambda k: self.children[k].visit_count)
            self.children[top] = _Child(child_visits[top], leader_breadth)


def _snapshot(depth0, depth1, depth2plus):
    """Minimal tracer snapshot shape: per-depth eligible-event counts."""
    return {"by_shape": {"c4a05": {
        "0": {"eligible_events": depth0},
        "1": {"eligible_events": depth1},
        "2+": {"eligible_events": depth2plus},
        "overall": {"eligible_events": depth0 + depth1 + depth2plus},
    }}}


def test_exactly_five_frozen_features():
    assert len(FEATURE_NAMES) == 5
    assert set(FEATURE_NAMES) == {
        "one_visit_backup_share", "depth3plus_backup_fraction",
        "leader_visit_margin", "root_policy_entropy", "leader_breadth"}


def test_features_are_computed_from_the_root_and_the_snapshot():
    root = _Root(priors={1: 0.5, 2: 0.3, 3: 0.2},
                 child_visits={1: 200, 2: 100, 3: 1}, leader_breadth=17)
    f = collect_boundary_features(root, _snapshot(100, 120, 80))
    assert set(f) == set(FEATURE_NAMES)
    # one child of three has exactly one visit
    assert f["one_visit_backup_share"] == pytest.approx(1 / 3)
    # depth 2+ is the proxy for backups reaching depth three or deeper
    assert f["depth3plus_backup_fraction"] == pytest.approx(80 / 300)
    # (200 - 100) / 301
    assert f["leader_visit_margin"] == pytest.approx(100 / 301)
    assert f["leader_breadth"] == 17
    h = -sum(p * math.log(p) for p in (0.5, 0.3, 0.2)) / math.log(3)
    assert f["root_policy_entropy"] == pytest.approx(h)


def test_empty_root_yields_None_not_zero():
    f = collect_boundary_features(_Root({}, {}), _snapshot(0, 0, 0))
    for k in ("one_visit_backup_share", "leader_visit_margin",
              "root_policy_entropy", "depth3plus_backup_fraction"):
        assert f[k] is None, f"{k} must be None when undefined, never 0.0"


def test_a_single_child_has_no_margin():
    f = collect_boundary_features(_Root({1: 1.0}, {1: 50}), _snapshot(10, 0, 0))
    assert f["leader_visit_margin"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_a.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/atlas_readout_a.py
"""Atlas Read-out A -- design section 6, FROZEN.

Five features at the 320-completion prefix, a fixed-ridge logistic classifier,
frozen validation bars, and the preregistered deployability aggregation.

Pure: every input is a plain object or dict, so this qualifies on synthetic rows.
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The five frozen features and ONLY these five. Section 6 forbids an
# unrestricted feature search; other tree statistics may be reported
# descriptively but must not enter the detector.
FEATURE_NAMES: Tuple[str, ...] = (
    "one_visit_backup_share",
    "depth3plus_backup_fraction",
    "leader_visit_margin",
    "root_policy_entropy",
    "leader_breadth",
)


def collect_boundary_features(root: Any, tracer_snapshot: Dict[str, Any]
                              ) -> Dict[str, Optional[float]]:
    """Features at the batch-safe boundary. Undefined -> None, never 0.0."""
    visited = {mv: c.visit_count for mv, c in root.children.items()
               if c.visit_count > 0}
    total = sum(visited.values())
    ordered = sorted(visited.values(), reverse=True)

    one_visit = (sum(1 for v in visited.values() if v == 1) / len(visited)
                 if visited else None)
    margin = ((ordered[0] - ordered[1]) / total
              if (len(ordered) >= 2 and total) else None)

    priors = [p for p in root.priors.values() if p > 0]
    entropy = None
    if len(priors) >= 2:
        s = sum(priors)
        norm = [p / s for p in priors]
        entropy = (-sum(p * math.log(p) for p in norm)) / math.log(len(norm))

    shape = next(iter(tracer_snapshot["by_shape"]))
    cells = tracer_snapshot["by_shape"][shape]
    overall = cells["overall"]["eligible_events"]
    deep = cells["2+"]["eligible_events"]
    depth_frac = (deep / overall) if overall else None

    breadth = None
    if visited:
        top_mv = max(visited, key=lambda k: (visited[k], -k))
        breadth = len(root.children[top_mv].children)

    return {
        "one_visit_backup_share": one_visit,
        "depth3plus_backup_fraction": depth_frac,
        "leader_visit_margin": margin,
        "root_policy_entropy": entropy,
        "leader_breadth": breadth,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_a.py -v -p no:cacheprovider`
Expected: PASS — 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_readout_a.py tests/test_atlas_readout_a.py
git commit -m "feat(atlas-s4): Read-out A boundary feature collector, five frozen features"
```

---

### Task 3: Read-out A — classifier, validation bars, deployability

**Files:**
- Modify: `scripts/GPU/alphazero/atlas_readout_a.py`
- Test: `tests/test_atlas_readout_a.py`

**Interfaces:**
- Consumes: `FEATURE_NAMES`, feature dicts, class labels.
- Produces: `fit_ridge_logistic(X, y, l2=1.0, iters=2000, lr=0.1) -> dict`; `standardize(rows, stats=None) -> tuple`; `auc(scores, labels) -> Optional[float]`; `bootstrap_auc_lower_bound(scores, labels, seed, replicates=10000) -> Optional[float]`; `evaluate_detector(discovery, validation) -> dict`; `deployability(remaining_values, strata=None) -> dict`.

Frozen §6 bars: validation holds **≥20 misleading and ≥25 stable-negative**; AUC **≥0.75**;
bootstrap lower bound **≥0.60**; a discovery-frozen threshold flags **≤25%** of validation
at precision **≥0.60**. Standardization is learned on **discovery only**.

Frozen §6 deployability: every `remaining == 0` row is non-actionable; **median
`remaining == 0` fails** the controller-deployability claim; report the zero-budget
fraction and remaining quartiles overall and by required strata, with **no**
stratum-specific gate.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_atlas_readout_a.py
from scripts.GPU.alphazero.atlas_readout_a import (
    auc, bootstrap_auc_lower_bound, deployability, evaluate_detector,
    fit_ridge_logistic, standardize,
)


def test_auc_is_one_for_perfect_separation_and_half_for_none():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == pytest.approx(0.5)


def test_auc_is_None_when_a_class_is_absent():
    assert auc([0.1, 0.2], [0, 0]) is None       # None, never 0.5 by default


def test_standardization_stats_come_from_discovery_only():
    disc = [{"a": 1.0}, {"a": 3.0}]
    _z, stats = standardize(disc, feature_names=("a",))
    val = [{"a": 5.0}]
    z_val, stats2 = standardize(val, feature_names=("a",), stats=stats)
    assert stats2 == stats                        # unchanged by validation data
    assert z_val[0][0] == pytest.approx((5.0 - 2.0) / stats["a"][1])


def test_ridge_logistic_separates_a_linearly_separable_set():
    X = [[-2.0], [-1.0], [1.0], [2.0]]
    y = [0, 0, 1, 1]
    model = fit_ridge_logistic(X, y)
    scores = [model["predict"](x) for x in X]
    assert auc(scores, y) == pytest.approx(1.0)


def test_detector_fails_closed_on_insufficient_validation_classes():
    r = evaluate_detector(discovery=[], validation=[])
    assert r["verdict"] == "INSUFFICIENT_CLASSES"
    assert r["auc"] is None and r["auc_lower_bound"] is None


def test_deployability_fails_when_the_MEDIAN_remaining_is_zero():
    r = deployability([0, 0, 0, 40, 60])
    assert r["median_remaining"] == 0
    assert r["verdict"] == "NOT_DEPLOYABLE"
    assert r["zero_budget_fraction"] == pytest.approx(3 / 5)


def test_deployability_passes_with_a_positive_median_and_reports_quartiles():
    r = deployability([10, 40, 60, 70, 80])
    assert r["verdict"] == "DEPLOYABLE"
    assert r["quartiles"] is not None and len(r["quartiles"]) == 3
    assert r["zero_budget_fraction"] == 0.0


def test_deployability_reports_strata_without_gating_on_them():
    r = deployability([0, 40, 60], strata={"late": [0, 0], "midgame": [60]})
    assert set(r["by_stratum"]) == {"late", "midgame"}
    # Section 6: no stratum-specific acceptance gate.
    assert "verdict" not in r["by_stratum"]["late"]
    assert r["by_stratum"]["late"]["median_remaining"] == 0


def test_deployability_of_an_empty_set_is_None_not_zero():
    r = deployability([])
    assert r["median_remaining"] is None
    assert r["zero_budget_fraction"] is None
    assert r["verdict"] == "NO_ROWS"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_a.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'auc'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/GPU/alphazero/atlas_readout_a.py
MIN_VALIDATION_MISLEADING = 20
MIN_VALIDATION_STABLE_NEGATIVE = 25
AUC_BAR = 0.75
AUC_LOWER_BOUND_BAR = 0.60
MAX_FLAG_RATE = 0.25
MIN_PRECISION = 0.60
BOOTSTRAP_REPLICATES = 10000


def standardize(rows: Sequence[Dict[str, Any]],
                feature_names: Sequence[str] = FEATURE_NAMES,
                stats: Optional[Dict[str, Tuple[float, float]]] = None):
    """Z-score. `stats` is learned on DISCOVERY only and passed to validation."""
    if stats is None:
        stats = {}
        for f in feature_names:
            vals = [r[f] for r in rows if r.get(f) is not None]
            mu = statistics.fmean(vals) if vals else 0.0
            sd = statistics.pstdev(vals) if len(vals) > 1 else 1.0
            stats[f] = (mu, sd if sd else 1.0)
    # Section 6a: a missing feature REJECTS the row. Imputing it to the
    # discovery mean would fabricate a maximally-uninformative observation at
    # the centre of the training distribution and silently dilute both classes.
    for i, r in enumerate(rows):
        missing = [f for f in feature_names if r.get(f) is None]
        if missing:
            raise ValueError(
                f"row {i} is missing features {missing}; rows with undefined "
                f"features are rejected, never imputed")
    X = [[(r[f] - stats[f][0]) / stats[f][1] for f in feature_names]
         for r in rows]
    return X, stats


def fit_ridge_logistic(X, y, l2: float = 1.0, iters: int = 2000,
                       lr: float = 0.1) -> Dict[str, Any]:
    """Closed-form-free gradient descent. stdlib only; no numpy or scipy."""
    n_f = len(X[0]) if X else 0
    w = [0.0] * n_f
    b = 0.0
    n = len(X) or 1
    for _ in range(iters):
        gw = [0.0] * n_f
        gb = 0.0
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
    """Rank AUC with ties at 0.5. None when a class is absent -- never a
    defaulted 0.5, which would look like a real chance result."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > q else 0.5 if p == q else 0.0)
               for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def bootstrap_auc_lower_bound(scores, labels, seed: int,
                              replicates: int = BOOTSTRAP_REPLICATES,
                              alpha: float = 0.05) -> Optional[float]:
    base = auc(scores, labels)
    if base is None:
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
                      seed: int = 20260804) -> Dict[str, Any]:
    """Section 6's frozen bars. Fails CLOSED on insufficient validation classes."""
    v_pos = sum(1 for r in validation if r["label"] == 1)
    v_neg = sum(1 for r in validation if r["label"] == 0)
    if v_pos < MIN_VALIDATION_MISLEADING or v_neg < MIN_VALIDATION_STABLE_NEGATIVE:
        return {"verdict": "INSUFFICIENT_CLASSES", "auc": None,
                "auc_lower_bound": None, "n_misleading": v_pos,
                "n_stable_negative": v_neg,
                "reason": "validation split cannot support its own gate"}

    Xd, stats = standardize(discovery)
    yd = [r["label"] for r in discovery]
    model = fit_ridge_logistic(Xd, yd)
    Xv, _ = standardize(validation, stats=stats)
    yv = [r["label"] for r in validation]
    sv = [model["predict"](x) for x in Xv]

    a = auc(sv, yv)
    lb = bootstrap_auc_lower_bound(sv, yv, seed=seed)
    # Threshold frozen on DISCOVERY, then applied to validation unchanged.
    sd = sorted((model["predict"](x) for x in Xd), reverse=True)
    thr = sd[max(0, int(MAX_FLAG_RATE * len(sd)) - 1)] if sd else 1.0
    flagged = [(s, y) for s, y in zip(sv, yv) if s >= thr]
    flag_rate = len(flagged) / len(sv) if sv else None
    precision = (sum(y for _s, y in flagged) / len(flagged)) if flagged else None

    passed = (a is not None and a >= AUC_BAR
              and lb is not None and lb >= AUC_LOWER_BOUND_BAR
              and flag_rate is not None and flag_rate <= MAX_FLAG_RATE
              and precision is not None and precision >= MIN_PRECISION)
    return {"verdict": "PASS" if passed else "FAIL", "auc": a,
            "auc_lower_bound": lb, "flag_rate": flag_rate,
            "precision": precision, "threshold": thr,
            "n_misleading": v_pos, "n_stable_negative": v_neg}


def deployability(remaining_values: Sequence[int],
                  strata: Optional[Dict[str, Sequence[int]]] = None
                  ) -> Dict[str, Any]:
    """Section 6's preregistered rule: every remaining == 0 row is
    non-actionable, and a MEDIAN of zero fails the controller-deployability
    claim. Strata are REPORTED, never gated -- a median rule catches tail-only
    and majority-tail behaviour without inventing a minimum second-stage budget.
    """
    def summarize(vals: Sequence[int]) -> Dict[str, Any]:
        if not vals:
            return {"n": 0, "median_remaining": None,
                    "zero_budget_fraction": None, "quartiles": None}
        s = sorted(vals)
        return {
            "n": len(s),
            "median_remaining": statistics.median(s),
            "zero_budget_fraction": sum(1 for v in s if v == 0) / len(s),
            "quartiles": (statistics.quantiles(s, n=4, method="inclusive")
                          if len(s) >= 2 else None),
        }

    overall = summarize(remaining_values)
    if overall["median_remaining"] is None:
        verdict = "NO_ROWS"
    elif overall["median_remaining"] == 0:
        verdict = "NOT_DEPLOYABLE"
    else:
        verdict = "DEPLOYABLE"
    return {**overall, "verdict": verdict,
            "by_stratum": {k: summarize(v) for k, v in (strata or {}).items()}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_a.py -v -p no:cacheprovider`
Expected: PASS — 13 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_readout_a.py tests/test_atlas_readout_a.py
git commit -m "feat(atlas-s4): Read-out A classifier, frozen bars and deployability rule"
```

---

### Task 4: Read-out B — four-rung gate calibration

**Files:**
- Create: `scripts/GPU/alphazero/atlas_readout_b.py`
- Test: `tests/test_atlas_readout_b.py`

**Interfaces:**
- Consumes: Stage 3 `LegResult` rows at all four rungs, plus the stable-reference result.
- Produces: `gate_triggers(legs, hi=1600) -> dict`; `closes_half(...)`; `convergent(legs, ref) -> dict`; `calibrate_gate(rows, gate_name) -> dict`; **`natural_convergence_report(rows) -> dict`** (the 400→6,400 reference distribution); **`compound_narrowing(legs) -> Optional[bool]`**; **`by_stratum_summary(rows, gate_name) -> dict`**.

> `gate_triggers` takes the upper rung as a parameter so the **400→6,400** report uses
> the same code as 400→1,600 — §7 requires both, and the 6,400 changes are the
> natural-convergence reference distribution. They are **reported, never used as
> causal evidence** that a same-budget intervention is safe.
>
> `compound_narrowing` returns `None` where it does not apply, never `False` — §7 says
> "where applicable", and an inapplicable condition is not a negative one.
>
> `by_stratum_summary` reports overall plus late / flat-policy / near-even. **No
> per-stratum acceptance gate** is created.

Frozen §7. The historical metrics are computed at **all four rungs** — the 3,200 rung is
required because distribution convergence checks **both** deep rungs for the **same**
metric. The gate denominator is `eligible_triggers` (triggers on stable-reference rows),
and the rule is ≥10 eligible, ≥75% convergent, **and** ≥15 percentage points above the
base convergent rate. The outcome is *needs review*, never *invalid*.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_readout_b.py
import pytest

from scripts.GPU.alphazero.atlas_labelling import stable_reference
from scripts.GPU.alphazero.atlas_readout_b import (
    BASE_RATE_MARGIN, MIN_ELIGIBLE_TRIGGERS, MIN_CONVERGENT_RATE,
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


def _row(v=(0.9, 0.4, 0.05, 0.05), m=(7, 3, 3, 3), shares=(0.5, 0.6, 0.7, 0.7),
         effs=(20.0, 16.0, 12.0, 12.0), ranks=(1, 1, 1, 1)):
    return [_leg(b, v[i], m[i], shares[i], effs[i], ranks[i])
            for i, b in enumerate((400, 1600, 3200, 6400))]


def test_frozen_thresholds_are_pinned():
    assert MIN_ELIGIBLE_TRIGGERS == 10
    assert MIN_CONVERGENT_RATE == 0.75
    assert BASE_RATE_MARGIN == 0.15


def test_closes_half_needs_a_real_gap_and_half_closure():
    assert closes_half(m400=1.0, m1600=0.4, D=0.0) is True      # 0.4 <= 0.5
    assert closes_half(m400=1.0, m1600=0.8, D=0.0) is False
    # No gap to begin with: vacuous, so it must NOT fire.
    assert closes_half(m400=0.5, m1600=0.5, D=0.5) is False


def test_convergent_requires_persistence_as_a_JOINT_condition():
    legs = _row()
    ref = stable_reference(legs)
    r = convergent(legs, ref)
    assert r["persistent"] is True and r["convergent"] is True
    # Break persistence: 1,600 picks a move the deep rungs do not keep.
    broken = _row(m=(7, 9, 3, 3))
    r2 = convergent(broken, stable_reference(broken))
    assert r2["persistent"] is False and r2["convergent"] is False


def test_dist_convergent_needs_the_SAME_metric_at_both_deep_rungs():
    """Checking 6,400 alone would let a distribution match a single unstable
    deep reading and score as convergence."""
    legs = _row(v=(0.06, 0.05, 0.05, 0.05), m=(3, 3, 3, 3),
                shares=(0.20, 0.60, 0.62, 0.62))
    r = convergent(legs, stable_reference(legs))
    assert r["dist_convergent"] is True
    mixed = _row(v=(0.06, 0.05, 0.05, 0.05), m=(3, 3, 3, 3),
                 shares=(0.20, 0.60, 0.90, 0.62))
    r2 = convergent(mixed, stable_reference(mixed))
    assert r2["dist_convergent"] is False


def test_gate_triggers_are_computed_at_all_four_rungs():
    legs = _row(shares=(0.90, 0.96, 0.97, 0.97))
    t = gate_triggers(legs)
    assert set(t) >= {"new_collapse", "lower_prior_flip",
                      "effective_children_drop", "top_share_increase"}
    assert t["new_collapse"] is True         # crossed 0.95 between 400 and 1600


def test_lower_prior_flip_uses_the_prior_RANK():
    legs = _row(m=(3, 9, 9, 9), ranks=(1, 7, 7, 7))
    assert gate_triggers(legs)["lower_prior_flip"] is True


def test_calibration_uses_the_ELIGIBLE_denominator():
    """Triggers on rows without a stable reference cannot be classified, so
    counting them as non-convergent would depress the rate."""
    rows = [_row() for _ in range(12)] + [_row(m=(7, 3, 3, 9))] * 5   # unstable
    r = calibrate_gate(rows, "top_share_increase")
    assert r["eligible_triggers"] <= r["total_triggers"]
    assert r["eligible_trigger_fraction"] is not None


def test_needs_review_requires_all_three_conditions():
    """Each condition is falsified individually; accepting either verdict would
    prove nothing."""
    conv = [_row() for _ in range(12)]              # convergent, gate fires
    # 1. too few eligible triggers
    assert calibrate_gate(conv[:5], "top_share_increase")["verdict"] == "no finding"
    # 2. convergent rate below 0.75 -- half the rows fail persistence
    mixed = conv[:6] + [_row(m=(7, 9, 3, 3)) for _ in range(6)]
    r_mixed = calibrate_gate(mixed, "top_share_increase")
    assert (r_mixed["convergent_rate"] or 0) < 0.75
    assert r_mixed["verdict"] == "no finding"
    # 3. base-rate margin: a gate firing on everything convergent gains nothing
    r_all = calibrate_gate(conv, "top_share_increase")
    if r_all["convergent_rate"] is not None and r_all["base_convergent_rate"] is not None:
        margin = r_all["convergent_rate"] - r_all["base_convergent_rate"]
        assert (r_all["verdict"] == "needs review") == (
            r_all["eligible_triggers"] >= 10 and r_all["convergent_rate"] >= 0.75
            and margin >= 0.15)
    assert "invalid" not in r_all["verdict"]         # never "invalid"


def test_natural_convergence_report_covers_400_to_6400():
    rows = [_row() for _ in range(4)]
    r = natural_convergence_report(rows)
    assert set(r) >= {"new_collapse", "top_share_increase"}
    # Reported as the reference distribution, NOT as causal evidence that a
    # same-budget intervention is safe.
    assert r["is_causal_evidence"] is False


def test_compound_narrowing_is_None_where_inapplicable():
    assert compound_narrowing(_row(shares=(None, None, None, None))) is None


def test_stratum_summary_reports_without_gating():
    rows = [_row() for _ in range(4)]
    s = by_stratum_summary(rows, "top_share_increase")
    assert "overall" in s
    for k, v in s.items():
        assert "verdict" not in v or k == "overall"


def test_calibration_is_None_not_zero_with_no_eligible_triggers():
    rows = [_row(m=(7, 3, 3, 9))] * 4        # no stable reference anywhere
    r = calibrate_gate(rows, "new_collapse")
    assert r["convergent_rate"] is None
    assert r["verdict"] == "no finding"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_b.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/atlas_readout_b.py
"""Atlas Read-out B -- design section 7, FROZEN.

Calibration, not a hypothesis: does an inherited collateral gate fire on changes
that move TOWARD the stable deeper reference?

The outcome is "needs review", never "invalid". Higher-budget fidelity is itself
only a proxy, while the old gates protect against collateral behaviour it does
not measure.
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


def _by_b(legs: Sequence[Any]) -> Dict[int, Any]:
    return {l.nominal_B: l for l in legs}


def gate_triggers(legs: Sequence[Any]) -> Dict[str, bool]:
    """The historical metrics, evaluated on the 400 -> 1,600 transition.

    Computed with all four rungs available, because distribution convergence
    below checks both deep rungs.
    """
    d = _by_b(legs)
    a, b = d[400], d[1600]
    collapse = (a.top_share is not None and b.top_share is not None
                and a.top_share < COLLAPSE_TOP_SHARE
                and b.top_share >= COLLAPSE_TOP_SHARE)
    flip = (a.selected_move != b.selected_move
            and a.selected_move_prior_rank is not None
            and b.selected_move_prior_rank is not None
            and b.selected_move_prior_rank > a.selected_move_prior_rank)
    eff_drop = (a.effective_children is not None
                and b.effective_children is not None
                and b.effective_children < a.effective_children)
    share_up = (a.top_share is not None and b.top_share is not None
                and b.top_share > a.top_share)
    return {"new_collapse": collapse, "lower_prior_flip": flip,
            "effective_children_drop": eff_drop, "top_share_increase": share_up}


def closes_half(m400: Optional[float], m1600: Optional[float],
                D: Optional[float]) -> bool:
    """|m400 - D| > 0 AND |m1600 - D| <= 0.5 * |m400 - D|.

    The `> 0` guard matters: with no gap to begin with, "closes half" is vacuous
    and must not fire rather than firing trivially.
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
    move_conv = (d[400].selected_move != deep
                 and d[1600].selected_move == deep)
    value_conv = (abs(d[1600].root_value - d[6400].root_value)
                  <= abs(d[400].root_value - d[6400].root_value)
                  - VALUE_CONVERGENCE_TOL)
    # SAME metric toward BOTH deep rungs -- the disjunction is over metrics,
    # not over rungs. Mixing one metric's 3,200 agreement with another's 6,400
    # is not evidence.
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


def calibrate_gate(rows: Sequence[Sequence[Any]], gate_name: str
                   ) -> Dict[str, Any]:
    """Section 7's frozen "needs review" rule, on the ELIGIBLE denominator."""
    total_triggers = eligible = confirmed = 0
    eligible_rows = base_convergent = 0
    for legs in rows:
        ref = stable_reference(legs)
        fired = gate_triggers(legs)[gate_name]
        if fired:
            total_triggers += 1
        if not ref["stable"]:
            continue                       # unclassifiable: excluded, not counted
        eligible_rows += 1
        conv = convergent(legs, ref)["convergent"]
        base_convergent += 1 if conv else 0
        if fired:
            eligible += 1
            confirmed += 1 if conv else 0

    rate = (confirmed / eligible) if eligible else None
    base_rate = (base_convergent / eligible_rows) if eligible_rows else None
    needs_review = (eligible >= MIN_ELIGIBLE_TRIGGERS
                    and rate is not None and rate >= MIN_CONVERGENT_RATE
                    and base_rate is not None
                    and (rate - base_rate) >= BASE_RATE_MARGIN)
    return {
        "gate": gate_name,
        "total_triggers": total_triggers,
        "eligible_triggers": eligible,
        "eligible_trigger_fraction": ((eligible / total_triggers)
                                      if total_triggers else None),
        "confirmed_convergent": confirmed,
        "convergent_rate": rate,
        "base_convergent_rate": base_rate,
        # "needs review" means the gate structure must be reviewed and frozen
        # before it judges another prototype. It does NOT mean the gate is
        # invalid and does not authorize deleting or relaxing it.
        "verdict": "needs review" if needs_review else "no finding",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_b.py -v -p no:cacheprovider`
Expected: PASS — 9 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_readout_b.py tests/test_atlas_readout_b.py
git commit -m "feat(atlas-s4): Read-out B four-rung gate calibration with eligible denominator"
```

---

### Task 5: Read-out C — retention, strata, shape selection

**Files:**
- Create: `scripts/GPU/alphazero/atlas_readout_c.py`
- Test: `tests/test_atlas_readout_c.py`

**Interfaces:**
- Consumes: Stage 3's two tracer snapshots and Stage 1's `k_of_n` / `n_admit`.
- Produces: `static_retention(...)`; `intervention_from_snapshots(...)`; **`classify_strata(row) -> set[str]`**; **`aggregate_shape(rows, shape) -> dict`**; `select_shape(per_shape) -> dict`; **`select_on_discovery_validate_on_selected(discovery, validation) -> dict`**; `STRATA`.

> **The row-to-aggregate path is the point.** `select_shape` consumes rates; nothing
> derived them from per-row results, which is why `MISLEADING_INTERVENTION_BAR` was
> unused. `aggregate_shape` now folds per-row three-valued results into the four
> rates, deriving root / depth-1 / two-ply retention from the 3,200 and 6,400
> reference lines and classifying the frozen strata. **Inconclusive rows are excluded
> from the intervention denominator and counted separately** — folding them in as
> either outcome would invent a measurement. If the denominator empties, the shape's
> rate is `None` and it **cannot pass**, rather than defaulting.
>
> Selection happens on **discovery**; only the selected shape is evaluated on
> **validation**, so a shape cannot be chosen for looking good on the split that
> judges it.

Frozen §8 bar: retain **≥95%** of stable deep root moves, **≥90%** of stable depth-1
replies, intervene on **≥50%** of misleading roots and **≤25%** of stable-negative roots.
**Lexicographic selection**: retention floors → stable-root intervention ceiling → higher
intervention on misleading → tie broken by higher descendant retention.

**Lag is directional.** Retention is evaluated under `K(n)`; the intervention threshold
must **also** pass under `K(n+14)`. Passing only under `K(n)` is **inconclusive**, not a
pass — the lag is conservative for retention and anti-conservative for intervention.

> **Incomplete in this revision — must be written before Task 5 is executed.**
> The Interfaces block above declares `classify_strata`, `aggregate_shape` and
> `select_on_discovery_validate_on_selected`, and the design note states what they
> must do, but their **implementations and tests are not yet written**. Task 5 is
> not executable until they are. Recorded here rather than left implicit, because a
> declared-but-absent function is exactly the shape of defect this plan keeps
> catching.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_readout_c.py
import pytest

from scripts.GPU.alphazero.atlas_readout_c import (
    RETENTION_DEPTH1_BAR, RETENTION_ROOT_BAR, STRATA,
    intervention_from_snapshots, select_shape, static_retention,
)
from scripts.GPU.alphazero.selection_tracer import WIDENING_SHAPES


def _priors(n, best):
    """n moves; `best` gets the top prior."""
    d = {i: (1.0 if i == best else 0.5 - i * 1e-4) for i in range(n)}
    return d


def test_frozen_bars_and_strata_are_pinned():
    assert RETENTION_ROOT_BAR == 0.95
    assert RETENTION_DEPTH1_BAR == 0.90
    assert set(STRATA) == {"late", "near_even", "root_flat",
                           "locally_flat_depth1", "locally_flat_depth2"}


def test_static_retention_admits_a_top_prior_move():
    r = static_retention(_priors(500, best=0), required_moves=[0],
                         n_at_selection=400, shape=("c4a05", 4.0, 0.5))
    assert r["retained"] == 1 and r["rate"] == pytest.approx(1.0)


def test_static_retention_rejects_a_deep_tail_move():
    r = static_retention(_priors(500, best=0), required_moves=[499],
                         n_at_selection=400, shape=("c4a05", 4.0, 0.5))
    assert r["retained"] == 0 and r["rate"] == 0.0


def test_static_retention_of_nothing_is_None_not_zero():
    r = static_retention(_priors(10, 0), required_moves=[],
                         n_at_selection=400, shape=("c4a05", 4.0, 0.5))
    assert r["rate"] is None


def _snap(outside, first_touch, first_touch_outside, eligible):
    return {"by_shape": {"c4a05": {"overall": {
        "eligible_events": eligible, "outside_events": outside,
        "first_touch_events": first_touch,
        "first_touch_outside_events": first_touch_outside,
        "excluded_prior_mass": 0.4,
    }}}}


def test_intervention_requires_the_LAGGED_bound_too():
    """Passing only under K(n) is inconclusive: the lag over-estimates
    intervention, so a marginal pass may be an artifact."""
    r = intervention_from_snapshots(
        {"at_boundary": _snap(30, 100, 12, 200)}, shape_key="c4a05",
        lagged_first_touch_outside=8)          # 8/100 = 0.08 < 0.10
    assert r["meaningfully_affected"] is None
    assert r["verdict"] == "INCONCLUSIVE"


def test_intervention_passes_when_both_bounds_clear():
    r = intervention_from_snapshots(
        {"at_boundary": _snap(30, 100, 15, 200)}, shape_key="c4a05",
        lagged_first_touch_outside=12)         # 0.12 >= 0.10
    assert r["meaningfully_affected"] is True
    assert r["verdict"] == "OK"


def test_shape_selection_is_lexicographic():
    a = {"root_retention": 0.99, "depth1_retention": 0.95,
         "misleading_intervention": 0.60, "stable_intervention": 0.20,
         "descendant_retention": 0.80}
    b = {"root_retention": 0.99, "depth1_retention": 0.95,
         "misleading_intervention": 0.55, "stable_intervention": 0.10,
         "descendant_retention": 0.99}
    # Both pass the floors and the ceiling; A wins on misleading intervention.
    assert select_shape({"c4a05": a, "c13a03": b})["selected"] == "c4a05"


def test_shape_failing_a_retention_floor_is_excluded_however_good_otherwise():
    a = {"root_retention": 0.90, "depth1_retention": 0.95,      # below 0.95
         "misleading_intervention": 0.99, "stable_intervention": 0.01,
         "descendant_retention": 0.99}
    b = {"root_retention": 0.96, "depth1_retention": 0.91,
         "misleading_intervention": 0.51, "stable_intervention": 0.24,
         "descendant_retention": 0.70}
    assert select_shape({"c4a05": a, "c13a03": b})["selected"] == "c13a03"


def test_no_shape_passing_is_a_named_failure_not_a_fallback():
    bad = {"root_retention": 0.10, "depth1_retention": 0.10,
           "misleading_intervention": 0.99, "stable_intervention": 0.99,
           "descendant_retention": 0.10}
    r = select_shape({"c4a05": bad, "c13a03": bad})
    assert r["selected"] is None
    assert r["verdict"] == "NO_SHAPE_PASSES"


def test_ties_break_on_descendant_retention():
    a = {"root_retention": 0.99, "depth1_retention": 0.95,
         "misleading_intervention": 0.60, "stable_intervention": 0.20,
         "descendant_retention": 0.70}
    b = {"root_retention": 0.99, "depth1_retention": 0.95,
         "misleading_intervention": 0.60, "stable_intervention": 0.20,
         "descendant_retention": 0.90}
    assert select_shape({"c4a05": a, "c13a03": b})["selected"] == "c13a03"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_c.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/atlas_readout_c.py
"""Atlas Read-out C -- design section 8, FROZEN.

Counterfactual COVERAGE analysis. It cannot prove progressive widening would
improve search, because applying widening changes the later tree.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from .selection_tracer import (
    MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE, WIDENING_SHAPES, k_of_n,
)

RETENTION_ROOT_BAR = 0.95
RETENTION_DEPTH1_BAR = 0.90
MISLEADING_INTERVENTION_BAR = 0.50
STABLE_INTERVENTION_CEILING = 0.25
BATCH_LAG = 14                     # eval_batch_size

# Frozen strata. Flat-policy status is recomputed LOCALLY along the reference
# line, not inherited from the root.
STRATA = ("late", "near_even", "root_flat",
          "locally_flat_depth1", "locally_flat_depth2")


def static_retention(root_priors: Dict[int, float],
                     required_moves: Sequence[int], n_at_selection: int,
                     shape: Tuple[str, float, float]) -> Dict[str, Any]:
    """Would the moves stable deeper search requires have been admitted?

    A static tree check using the same frozen rank rule; no selection-event dump
    is needed. Retention is evaluated under K(n) -- the narrower, conservative
    admitted set -- so a pass here is genuinely safe.
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
    """Meaningful intervention, with the DIRECTIONAL lag bound.

    Completed visits lag in-flight work by up to one batch, so K(n) is narrower
    than a real implementation's: conservative for retention, ANTI-conservative
    for intervention. The threshold must therefore also pass under K(n+14).
    Passing only under K(n) is INCONCLUSIVE, not a pass.
    """
    cell = snapshots["at_boundary"]["by_shape"][shape_key]["overall"]
    ft = cell["first_touch_events"]
    rate = (cell["first_touch_outside_events"] / ft) if ft else None
    # PRODUCED by the tracer under K(n+14), never supplied by a caller.
    lagged_rate = ((cell["lagged_first_touch_outside_events"] / ft)
                   if ft else None)

    if rate is None:
        return {"first_touch_outside_rate": None, "lagged_rate": None,
                "meaningfully_affected": None, "verdict": "NO_EVENTS"}
    passes = rate >= MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE
    lagged_passes = (lagged_rate is not None
                     and lagged_rate >= MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE)
    if passes and lagged_passes:
        return {"first_touch_outside_rate": rate, "lagged_rate": lagged_rate,
                "meaningfully_affected": True, "verdict": "OK"}
    if passes and not lagged_passes:
        # None, never False: this is undecided, not a measured negative.
        return {"first_touch_outside_rate": rate, "lagged_rate": lagged_rate,
                "meaningfully_affected": None, "verdict": "INCONCLUSIVE"}
    return {"first_touch_outside_rate": rate, "lagged_rate": lagged_rate,
            "meaningfully_affected": False, "verdict": "OK"}


def select_shape(per_shape: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Frozen LEXICOGRAPHIC order (design section 8):

      1. root and reference-reply retention floors must pass
      2. the stable-root intervention ceiling must pass
      3. among survivors, higher intervention on misleading roots
      4. exact tie: higher descendant reference retention
    """
    survivors = {
        k: v for k, v in per_shape.items()
        if v["root_retention"] >= RETENTION_ROOT_BAR
        and v["depth1_retention"] >= RETENTION_DEPTH1_BAR
        and v["stable_intervention"] <= STABLE_INTERVENTION_CEILING
    }
    if not survivors:
        return {"selected": None, "verdict": "NO_SHAPE_PASSES",
                "considered": sorted(per_shape)}
    best = max(survivors,
               key=lambda k: (survivors[k]["misleading_intervention"],
                              survivors[k]["descendant_retention"]))
    return {"selected": best, "verdict": "OK",
            "survivors": sorted(survivors), "considered": sorted(per_shape)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_c.py -v -p no:cacheprovider`
Expected: PASS — 10 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_readout_c.py tests/test_atlas_readout_c.py
git commit -m "feat(atlas-s4): Read-out C retention, lag bound and lexicographic selection"
```

---

### Task 6: Artifact schema, provenance validation and `_jsonable`

**Files:**
- Create: `scripts/GPU/alphazero/atlas_artifact.py`
- Test: `tests/test_atlas_artifact.py`

**Interfaces:**
- Consumes: everything above, plus `_jsonable` from `build_atlas_corpus`.
- Produces: `ROW_SCHEMA_VERSION`; `build_row(...) -> dict`; `validate_provenance(run) -> dict`; `emit(run) -> str`.

Every row must carry the §2b/§4 facts that a later reader cannot reconstruct:
inheritance resets, `remaining`, and every undefined value **as `None`**.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_artifact.py
import json

import pytest

from scripts.GPU.alphazero.atlas_artifact import (
    ROW_SCHEMA_VERSION, build_row, emit, validate_provenance,
)


def _row_kwargs(**over):
    base = dict(
        game_idx=3, replay_seed=20400003, target_ply=95, phase="late",
        side="black", split="validation", inherited_I=137,
        reset_count=1, reset_rate=0.02, last_reset_ply=44,
        boundary={"N_actual": 326, "overshoot": 6, "remaining": 74,
                  "flush_type": "full"},
        legs=[{"nominal_B": 400}], label="misleading",
        features={"one_visit_backup_share": 0.4},
        tracer_snapshots={"at_boundary": {}, "at_400": {}},
    )
    base.update(over)
    return base


def test_row_carries_resets_remaining_and_the_schema_version():
    r = build_row(**_row_kwargs())
    assert r["schema_version"] == ROW_SCHEMA_VERSION
    assert r["reset_count"] == 1 and r["reset_rate"] == 0.02
    assert r["last_reset_ply"] == 44
    assert r["boundary"]["remaining"] == 74


def test_undefined_values_stay_None_through_emission():
    r = build_row(**_row_kwargs(reset_rate=None, last_reset_ply=None,
                                boundary=None))
    text = emit({"rows": [r], "provenance": {}})
    back = json.loads(text)["rows"][0]
    assert back["reset_rate"] is None
    assert back["last_reset_ply"] is None
    assert back["boundary"] is None          # never {} and never 0


def test_a_row_missing_the_boundary_is_flagged_not_defaulted():
    r = build_row(**_row_kwargs(boundary=None))
    assert r["boundary_missing"] is True


def test_emission_goes_through_jsonable():
    """Tuple-keyed payloads must survive; json.dumps cannot use tuple keys and
    `default=` rescues only values."""
    run = {"rows": [], "provenance": {},
           "cells": {("discovery", "late", "red"): 12}}
    back = json.loads(emit(run))
    assert back["cells"] == {"discovery|late|red": 12}


def test_provenance_validation_fails_closed_on_a_dirty_tree():
    r = validate_provenance({"git_head": "a" * 40, "worktree_clean": False,
                             "checkpoint_sha1": "0" * 40})
    assert r["verdict"] == "PROVENANCE_FAILURE"
    assert "worktree_clean" in r["problems"]


def test_provenance_validation_requires_a_checkpoint_digest():
    r = validate_provenance({"git_head": "a" * 40, "worktree_clean": True,
                             "checkpoint_sha1": ""})
    assert r["verdict"] == "PROVENANCE_FAILURE"
    assert "checkpoint_sha1" in r["problems"]


def test_valid_provenance_passes():
    r = validate_provenance({"git_head": "a" * 40, "worktree_clean": True,
                             "checkpoint_sha1": "0" * 40})
    assert r["verdict"] == "OK" and r["problems"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_artifact.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/atlas_artifact.py
"""Atlas artifact schema, provenance validation and emission.

Every undefined value stays None through emission. A missing boundary is FLAGGED,
never defaulted -- a zero-filled record is indistinguishable from a real one.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .build_atlas_corpus import _jsonable

ROW_SCHEMA_VERSION = 1


def build_row(*, game_idx: int, replay_seed: int, target_ply: int, phase: str,
              side: str, split: str, inherited_I: int, reset_count: int,
              reset_rate: Optional[float], last_reset_ply: Optional[int],
              boundary: Optional[Dict[str, Any]], legs: List[Dict[str, Any]],
              label: str, features: Dict[str, Any],
              tracer_snapshots: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "game_idx": game_idx, "replay_seed": replay_seed,
        "target_ply": target_ply, "phase": phase, "side": side, "split": split,
        "inherited_I": inherited_I,
        # Section 2b: reset statistics are explicit, and every row is kept.
        "reset_count": reset_count, "reset_rate": reset_rate,
        "last_reset_ply": last_reset_ply,
        "boundary": boundary, "boundary_missing": boundary is None,
        "legs": legs, "label": label, "features": features,
        "tracer_snapshots": tracer_snapshots,
    }


def validate_provenance(prov: Dict[str, Any]) -> Dict[str, Any]:
    """Fails CLOSED. A dirty tree or an unidentifiable checkpoint means the run
    is not reconstructible, whatever its numbers say."""
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
    """Serialize through _jsonable -- geometry and ladder types keep their
    natural tuples, and only the boundary normalizes."""
    # No default=str: it would stringify a schema defect instead of rejecting
    # it, turning an unserializable object into a plausible-looking value.
    return json.dumps(_jsonable(run), indent=2, sort_keys=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_artifact.py -v -p no:cacheprovider`
Expected: PASS — 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_artifact.py tests/test_atlas_artifact.py
git commit -m "feat(atlas-s4): artifact schema, fail-closed provenance and jsonable emission"
```

---

### Task 7: End-to-end read-out chain and the full suite

**Files:**
- Create: `tests/test_atlas_readout_chain.py`

**Interfaces:**
- Consumes: all four modules.
- Produces: no new API — this drives real Stage 3 ladder output into the real read-outs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_readout_chain.py
"""Real Stage 3 ladder output -> real Stage 4 read-outs. No surrogates."""
import json
import random

from scripts.GPU.alphazero.atlas_artifact import build_row, emit
from scripts.GPU.alphazero.atlas_labelling import class_counts, classify_row
from scripts.GPU.alphazero.atlas_readout_a import collect_boundary_features
from scripts.GPU.alphazero.atlas_readout_b import calibrate_gate, gate_triggers
from scripts.GPU.alphazero.corpus_geometry import GameMeta
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.selection_tracer import SelectionTracer
from scripts.GPU.alphazero.warm_prefix_replay import (
    BatchSafeBoundaryObserver, replay_prefix, run_additive_ladder,
)

from tests.eval_fakes import FakeEvaluator

BASE = 20400000
SIZE = 6


def _cfg(n=1):
    return MCTSConfig(n_simulations=n, eval_batch_size=14,
                      stall_flush_sims=48, pending_virtual_visits=8)


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


def _real_row():
    hist = _history(4)
    meta = GameMeta(game_id=0, seed=BASE, n_moves=len(hist), start_player="red")
    m = MCTS(FakeEvaluator(value=0.0), _cfg(), random.Random(BASE))
    pre = replay_prefix(m, meta, hist, target_ply=2, active_size=SIZE)
    tracer = SelectionTracer()
    m._selection_observer = tracer
    # THE FROZEN INCREMENTS. Tiny ones give nominal budgets 8/16/24/32, which
    # labelling and gate calibration -- both of which index 400/1,600/3,200/6,400
    # -- reject outright, so the "real chain" would never reach an assertion.
    # 6,400 simulations of FakeEvaluator at active_size=6 is CPU-only and fast.
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I, tracer=tracer)
    legs, snaps = run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                                      boundary_observer=obs,
                                      target_tracer=tracer)   # frozen defaults
    return meta, pre, legs, snaps, obs, tracer


def test_real_legs_classify_without_error():
    _meta, _pre, legs, _snaps, _obs, _tr = _real_row()
    label = classify_row(legs)
    assert label in {"misleading", "stable_negative", "ambiguous",
                     "no_stable_reference"}
    assert set(class_counts([legs])) >= {"misleading", "stable_negative"}


def test_real_gate_triggers_and_calibration_run_on_real_legs():
    _meta, _pre, legs, _snaps, _obs, _tr = _real_row()
    t = gate_triggers(legs)
    assert set(t) == {"new_collapse", "lower_prior_flip",
                      "effective_children_drop", "top_share_increase"}
    r = calibrate_gate([legs], "top_share_increase")
    assert r["verdict"] in {"needs review", "no finding"}


def test_real_boundary_features_are_collected_from_the_real_tree():
    _meta, pre, _legs, snaps, _obs, _tr = _real_row()
    f = collect_boundary_features(
        capture_start=snaps["features_at_start"],
        capture_boundary=snaps["features_at_boundary"],
        n_actual=obs.record.N_actual, root_priors=pre.root.priors,
        leader_breadth=snaps["features_at_boundary"]["n_visited_children"])
    assert set(f) == {"one_visit_backup_share", "depth3plus_backup_fraction",
                      "leader_visit_margin", "root_policy_entropy",
                      "leader_breadth"}


def test_a_real_row_survives_the_artifact_boundary():
    meta, pre, legs, snaps, obs, _tr = _real_row()
    row = build_row(
        game_idx=meta.game_id, replay_seed=meta.seed, target_ply=2,
        phase="opening", side="red", split="discovery",
        inherited_I=pre.inherited_I, reset_count=pre.reset_count,
        reset_rate=pre.reset_rate, last_reset_ply=pre.last_reset_ply,
        boundary=(vars(obs.record) if obs.record else None),
        legs=[vars(l) for l in legs], label=classify_row(legs),
        features=collect_boundary_features(pre.root, snaps["at_boundary"]),
        tracer_snapshots=snaps)
    back = json.loads(emit({"rows": [row], "provenance": {}}))["rows"][0]
    assert back["inherited_I"] == pre.inherited_I
    assert len(back["legs"]) == 4
    assert back["boundary_missing"] is (obs.record is None)
```

- [ ] **Step 2: Run, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_chain.py -v -p no:cacheprovider`
Expected: PASS — 4 passed.

```bash
.venv/bin/python -m pytest -p no:cacheprovider -q > /tmp/s4.out 2>&1; echo "REAL_EXIT=$?" >> /tmp/s4.out; tail -3 /tmp/s4.out
```

Expected full suite: **2435 + 62 = 2497 passed**, 4 skipped, 0 failed.
Per file: artifact 7, labelling 10, producer_closure 6, readout_a 13, readout_b 12, readout_c 10, readout_chain 4. A different total means tests were added, lost or renamed —
investigate before committing; the delta is the qualification check.

- [ ] **Step 3: Commit**

```bash
git add tests/test_atlas_readout_chain.py
git commit -m "test(atlas-s4): real Stage 3 ladder output into the real read-outs"
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
- [ ] **62 new tests** (artifact 7, labelling 10, producer_closure 6, readout_a 13, readout_b 12, readout_c 10, readout_chain 4). Full suite **2497 passed**, exit code read from the process.

## Out of scope

No reservoir generation, no checkpoint loading, no MLX execution, no measurement run.
The three distribution gaps — real-scale throughput, the `remaining` distribution, and
the inheritance-reset rate — remain **operator/pilot measurements**. Stage 5 is planned
only after these interfaces exist and qualify.
