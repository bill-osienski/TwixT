# Atlas Stage 4 — Read-outs A / B / C and Artifact Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and qualify labelling, capacity sizing, all three read-outs, and the artifact schema — every part pure and testable on synthetic ladder output, with no reservoir, checkpoint, MLX or measurement.

**Architecture:** Four pure modules over Stage 3's `LegResult` / `BoundaryRecord` / tracer snapshots: `atlas_labelling.py` (stable reference, classes, capacity), `atlas_readout_a.py` (features at both frozen instants, ridge classifier, dual pipeline, deployability), `atlas_readout_b.py` (four-rung gate calibration), `atlas_readout_c.py` (edge retention, strata, shape selection, validation verdict). One `atlas_artifact.py` handles schema, provenance and `_jsonable` boundaries. Every input is a plain dataclass or dict, so the whole stage qualifies on synthetic rows.

## Revision 2 — 2026-08-04, rewritten against amendments 3 and 4

Revision 1 was drafted before spec amendment 4 and before the amendment-3 clauses on
`K(n)`, excluded mass, verdict precedence and `LATE_ONLY_SEPARATION` were frozen. Four
coupled sections were rewritten together, because amendment 4 changes the **shape** of
what crosses the producer/consumer seam and fixing one end alone leaves the other
describing a contract that no longer exists:

| # | Was | Now | Governed by |
|---|---|---|---|
| 1 | Task 0 summarized **one** line off the final 6,400 tree, so `root_effective_visits` was `I + 6400` and every reply count was a final-tree count | **Two** deep lines captured while each state exists, plus complete depth-two parent-visit maps at the boundary **and** `B = 400`, with priors equality asserted where both rungs captured the same parent | §6a amendment 4 |
| 2 | Read-out C walked one line's `root_priors` / `reply` / `two_ply` | `aggregate_shape` over the **deduplicated union** of `(parent path, move)` edges at both instants — depth buckets, excluded prior mass, forced-root bypasses, forced-simulation counts, per-stratum retention; `classify_edge_strata` wired in | §6a amendment 4, §8 |
| 3 | The validation aggregate was computed and never judged | Three-way verdict with frozen precedence `FAIL > INCONCLUSIVE > PASS` | §6a "Validation is judged, not merely computed" |
| 4 | `evaluate_detector` ran once | Identical pipeline on **both** feature sets, boundary authoritative, `LATE_ONLY_SEPARATION` = boundary `FAIL` **and** `B = 400` `PASS` | §6a "Read-out A runs on both feature sets" |

Also corrected here: the interface blocks that still said `reference_line` singular while
the chain test already read `reference_lines` plural; Task 5's interface block, which
omitted `classify_edge_strata` entirely although Task 5 defines it; and the per-file test
counts, which were wrong for Read-out B (claimed 12, the plan contains 14) and Read-out C
(claimed 14, the plan contains 16) before any of this rewrite. **No frozen parameter,
threshold or predicate changed.**

## Revision 3 — 2026-08-04, review fixes

Six defects found reviewing revision 2. Four of them are the same failure this line of
work keeps producing — a contract that reads correctly in prose and is not the one the
code implements.

| # | Defect | Fix |
|---|---|---|
| 1 | **The artifact could not feed Read-out C.** `build_row` stored the producer document as `tracer_snapshots` and *separately* copied `reference_lines` and `parent_visits` to the top level, while Read-out C consumes `row["snapshots"]`. The chain worked around it with a hand-built `_c_row` — recreating the surrogate seam this plan exists to eliminate, and leaving the artifact path untested. | One undivided `snapshots` document. An artifact row **is** a Read-out A row and a Read-out C row; the chain drives both from real `build_row` output and `_c_row` is gone. |
| 2 | **Unstable roots entered the retention bars.** Every row contributed required edges, including `no_stable_reference` rows whose deep rungs never agreed — but §8's floors are about *stable* deep moves. | Gated retention accumulates only over `STABLE_REFERENCE_LABELS` (an allow-list); selection-event counters still cover every row; `retention_rows` and `rows_without_stable_reference` are reported. |
| 3 | **Two missingness states were conflated.** `agree is None` covered both "present in one line" and "absent from both". | Four explicit states — `agree` / `disagree` / `single_line` / `absent_both`; the last two are counted separately and neither enters the denominator. |
| 4 | **Row overlap ignored discovery**, so two models fitted on different discovery rows could report identical row sets. Separately, the plan let a boundary missing-feature rejection sit inside a `LATE_ONLY_SEPARATION`, contradicting amendment 4's own text. | `row_overlap` reports discovery and validation separately. **The contradiction is resolved in favour of the amendment**: any boundary-set rejection blocks the lateness verdict, reported as the boundary's own `FAIL` plus `lateness_blocked_by`. See the reasoning under Task 3 — narrowing the amendment was the alternative and was rejected. |
| 5 | **Provenance was not fail-closed.** `emit` never called `validate_provenance`, and the chain proved an empty provenance object could be emitted. Digests were checked for length 40 only. | `emit` refuses to serialize a run whose provenance does not validate, before serializing so a payload defect still raises `TypeError`; digests must be hexadecimal. |
| 6 | Task 4's expected result said 12 against 14 present. | Corrected. |

Test count moved **113 → 119**; expected suite **2554**. Again a derivation, not a target.

## Revision 4 — 2026-08-04, second review pass

| # | Defect | Fix |
|---|---|---|
| 1 | **The `LATE_ONLY_SEPARATION` fixture could not pass.** Revision 3 grew it by three *misleading* rows — 23 of 83 — so a perfect `B = 400` detector flags 27.7% and busts the frozen 25% ceiling. The test asserted a `PASS` that was arithmetically unreachable. | `_dual_rows` takes class counts; the fixture is **21 / 63**. `B = 400` flags 21/84 = exactly 0.25, and rejecting one boundary misleading row leaves 20/83 — still capacity-valid, so the boundary genuinely `FAIL`s rather than going insufficient. |
| 2 | **Read-out B still bypassed the artifact**, rebuilding its row by hand and contradicting the criterion that every read-out consumes a real `build_row` result. The cause was a type mismatch: `build_row` stored `vars(l)`-flattened legs, while Read-out B and `atlas_labelling` read `l.nominal_B` by **attribute**. | The row keeps `LegResult` / `BoundaryRecord` **objects**, and `_jsonable` gains an additive dataclass branch — the same "normalize at the JSON boundary" rule it already applies to tuple keys. The chain hands `artifact_row` straight to `calibrate_gate`. |

Test count **unchanged at 119**: the fixture fix is a parameter change, the Read-out B fix
is a rewrite of an existing test, and the dataclass boundary is pinned by extending the
emission test rather than adding one. Fix 2 touches landed Stage 2 code, so
`tests/test_build_atlas_corpus_cli.py` is re-run and must stay at 13.

A **Stage 5 handoff** is now recorded in the completion criteria: `flat_policy` and
`near_even` are supplied facts here, hardcoded throughout the synthetic chain, and Stage 5
must derive them from the frozen measured fields and qualify that producer seam.

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
| `scripts/GPU/alphazero/atlas_readout_a.py` (create) | Features from the frozen captures, ridge classifier, validation bars, deployability, and the dual boundary/`B=400` pipeline with `LATE_ONLY_SEPARATION`. |
| `scripts/GPU/alphazero/atlas_readout_b.py` (create) | Four-rung historical metrics, frozen convergence predicate, eligible-trigger gate. |
| `scripts/GPU/alphazero/atlas_readout_c.py` (create) | Edge-union retention at both instants, per-stratum and depth-bucket aggregation, lag bound, lexicographic shape selection, three-way validation verdict. |
| `scripts/GPU/alphazero/atlas_artifact.py` (create) | Row/run schema, provenance validation, `_jsonable` emission. |
| `scripts/GPU/alphazero/warm_prefix_replay.py` (modify, Task 0) | Adds the frozen captures, both deep reference lines and both parent-visit maps. |
| `scripts/GPU/alphazero/selection_tracer.py` (modify, Task 0) | Adds the online `K(n+14)` lagged counters. |
| `tests/test_atlas_producer_closure.py`, `..._labelling.py`, `..._readout_a.py`, `..._readout_b.py`, `..._readout_c.py`, `..._artifact.py`, `..._readout_chain.py` (create) | One suite per module plus the real-chain suite; synthetic or `FakeEvaluator` inputs only. |

---

### Task 0: Producer closure — freeze what the pure analyses will consume

**Files:**
- Modify: `scripts/GPU/alphazero/warm_prefix_replay.py`
- Modify: `scripts/GPU/alphazero/selection_tracer.py`
- Test: `tests/test_atlas_producer_closure.py`

**Interfaces:**
- Consumes: Stage 3's `run_additive_ladder`, `BatchSafeBoundaryObserver`, `SelectionTracer`, and `mcts.visit_leader_move` — all delivered and unchanged.
- Produces, in `warm_prefix_replay.py`: `capture_tree_state(root) -> dict` (the **complete compact capture schema** below); `check_backup_invariant(d3_start, d3_boundary, n_actual) -> bool`; `capture_parent_visits(root, max_depth=2) -> dict[tuple[int, ...], int]`; `deep_reference_line(root) -> dict`; `merge_reference_lines(line_3200, line_6400) -> dict`.
- Produces, in `selection_tracer.py`: `BATCH_LAG = 14` and a per-cell `lagged_first_touch_outside_events` counted **in the same pass** as the unlagged one.
- Produces, from `run_additive_ladder`: the Stage 3 return shape `(legs, snapshots)` is kept and **extended additively** — `snapshots` keeps its delivered `at_boundary` / `at_400` tracer snapshots at top level and gains three sibling keys:

```python
{
  "at_boundary": <tracer snapshot>,      # Stage 3, unchanged
  "at_400":      <tracer snapshot>,      # Stage 3, unchanged
  "captures":       {"at_start": ..., "at_boundary": ..., "at_400": ...},
  "parent_visits":  {"at_boundary": ..., "at_400": ...},
  "reference_lines": {"at_3200": ..., "at_6400": ..., "merged": ...},
}
```

Additive, so every Stage 3 consumer and every already-green Stage 3 test keeps working
unchanged. `reference_lines` is **plural** everywhere — revision 1 declared it plural in
one place and singular in five others.

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

#### The reference-line contract (amendment 4), frozen here

Read-out C evaluates admission against the moves stable deeper search requires. Four
things follow, and all four are producer obligations:

**a. Two deep lines, each captured while its state exists.** `deep_reference_line(root)`
is called at the end of leg 3 (nominal `B = 3,200`) and again at the end of leg 4
(`B = 6,400`). The ladder is additive on one tree, so after leg 4 the 3,200 state exists
nowhere and a single post-ladder summary can only ever describe 6,400.

**b. Edges, not moves.** A line is a list of `(parent_path, move)` edges, where
`parent_path` is the tuple of move ids from the target root — `()` at the root, `(m0,)`
at the reply, `(m0, m1)` at two ply. The required set is the **deduplicated union** of
both lines' edges: agreements collapse to one entry, and **where the two rungs disagree
both edges are retained.** Neither line is truth; the 3,200/6,400 pair exists so that
agreement is a *finding*.

**c. Agreement is equality of the complete edge.** The same move id reached through a
different parent is a different edge, and scoring it as agreement would overstate how
much the two rungs concur. Each depth carries one of **four** states, and the last two
both sit **outside the agreement denominator**:

```text
agree        both lines reached this depth and the edges are identical
disagree     both lines reached it and the edges differ
single_line  exactly one line reached it   -- nothing to compare against
absent_both  neither line reached it       -- a DIFFERENT missingness state
```

`single_line` and `absent_both` are counted **separately**. Amendment 4 names only the
former; folding a depth neither line reached into it would report a comparison that was
never even half-available. Reply and two-ply agreement are reported separately and **add
no gate**.

**d. Parent visits are captured as COMPLETE MAPS at the two shallow instants.**
`capture_parent_visits` records `path -> visit_count` for every existing path through
depth two, at the boundary and again at `B = 400`. It must be a complete map rather than
a lookup of the line's paths, because **the union is not known until leg 4 and these
maps are taken during leg 1.** A path absent from a map has **zero** visits, and
`K(0) = min(n_legal, max(1, 0)) = 1` admits only rank 1 there.

Retention therefore never reads a 6,400-era visit count. To make that structurally
impossible rather than merely intended, `deep_reference_line` records each edge's
**parent priors and nothing else about the parent** — the field that caused the revision-1
defect (`root_effective_visits` off the final tree) is simply not produced. Priors are
what ranks require, and under the frozen `add_noise=False` ladder they do not change
between rungs, so where both lines captured the same **parent** the two prior maps are
**asserted equal** rather than assumed so.

> **Why this task exists, and why it is first.** Stage 4's analyses are pure, which
> made four producer gaps invisible until review:
>
> 1. `depth3plus_backup_fraction` was read off **selection events**, but those are edge
>    traversals — a depth-5 simulation emits five of them and is one backup. §6a now
>    freezes the two-point `D3` accounting instead.
> 2. Boundary features were computed from `pre.root` **after** the ladder, but the
>    ladder mutates that root in place through all four legs, so the values described
>    the 6,400 tree. They must be frozen at the boundary and at `B = 400`.
> 3. The `K(n+14)` lagged count was **caller-supplied**. No real row could produce it,
>    so the bound was untestable in practice.
> 4. The reference line was summarized **once, off the final 6,400 tree**, so the
>    horizon Read-out C measured against was `I + 6400` rather than the `I + N_actual`
>    and `I + 400` the widening rule would actually see. Amendment 4 now requires two
>    lines and two parent-visit maps, all captured while their states exist.
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
    BatchSafeBoundaryObserver, capture_parent_visits, capture_tree_state,
    deep_reference_line, merge_reference_lines, replay_prefix,
    run_additive_ladder,
)

from tests.eval_fakes import FakeEvaluator

BASE = 20400000
SIZE = 6


def _edge(path, move, priors):
    """A reference-line edge as `deep_reference_line` emits it."""
    return {"parent_path": path, "move": move, "depth": len(path),
            "parent_priors": priors}


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


# -- amendment 4: two deep lines, and complete parent-visit maps --------------

def test_parent_visits_map_covers_every_path_through_depth_two():
    """A COMPLETE map, not a lookup of the line's paths: the union of required
    edges is not known until leg 4, and this map is taken during leg 1."""
    root = _N(10, {0: _N(6, {0: _N(4, {0: _N(3, move=0)}, move=0)}, move=0),
                   1: _N(4, move=1)})
    pv = capture_parent_visits(root)
    assert pv[()] == 10                      # the root is a parent, at depth 0
    assert pv[(0,)] == 6 and pv[(1,)] == 4
    assert pv[(0, 0)] == 4
    # Depth three is below every possible parent of a two-ply edge.
    assert (0, 0, 0) not in pv


def test_a_deep_line_is_a_list_of_edges_with_parent_paths():
    """Edges, not moves: the same move id under a different parent is a
    different edge, which is what makes agreement checkable at all."""
    root = _N(10, {5: _N(8, {3: _N(6, {1: _N(4, move=1)}, move=3)}, move=5),
                   9: _N(2, move=9)})
    line = deep_reference_line(root)
    assert [(e["parent_path"], e["move"]) for e in line["edges"]] == [
        ((), 5), ((5,), 3), ((5, 3), 1)]
    assert [e["depth"] for e in line["edges"]] == [0, 1, 2]
    # Parent PRIORS ride along, because ranks need them. The parent's visit
    # count deliberately does NOT: it would be a 6,400-era number, and its
    # presence is what let revision 1 measure against the wrong horizon.
    assert line["edges"][0]["parent_priors"] == root.priors
    assert "parent_effective_visits" not in line["edges"][0]


def test_the_union_keeps_both_replies_when_the_deep_lines_disagree():
    """Neither deep rung is truth, so a disagreement retains BOTH edges."""
    root_p = {7: 0.6, 8: 0.4}
    reply_p = {1: 0.5, 2: 0.5}
    l32 = {"edges": [_edge((), 7, root_p), _edge((7,), 1, reply_p)]}
    l64 = {"edges": [_edge((), 7, root_p), _edge((7,), 2, reply_p)]}
    m = merge_reference_lines(l32, l64)
    assert [(e["parent_path"], e["move"]) for e in m["required_edges"]] == [
        ((), 7), ((7,), 1), ((7,), 2)]
    assert m["required_edges"][0]["sources"] == (3200, 6400)   # collapsed
    assert m["required_edges"][1]["sources"] == (3200,)
    assert m["agreement"]["reply"]["state"] == "disagree"


def test_agreement_is_equality_of_the_complete_edge_not_the_move_id():
    """Move id 1 sits at depth 1 in both lines, under DIFFERENT parents. Scoring
    that as agreement would overstate how much the two rungs concur."""
    root_p = {7: 0.6, 8: 0.4}
    reply_p = {1: 0.5, 2: 0.5}
    l32 = {"edges": [_edge((), 7, root_p), _edge((7,), 1, reply_p)]}
    l64 = {"edges": [_edge((), 8, root_p), _edge((8,), 1, reply_p)]}
    m = merge_reference_lines(l32, l64)
    assert m["agreement"]["root"]["state"] == "disagree"
    assert m["agreement"]["reply"]["state"] == "disagree"
    assert len(m["required_edges"]) == 4          # nothing collapsed


def test_single_line_and_absent_both_are_DIFFERENT_missingness_states():
    """Amendment 4 calls a pair present in only one line "counted separately".
    A depth neither line reached is a different fact -- collapsing the two
    would report a comparison that was never even half-available. Neither
    enters the agreement denominator."""
    root_p = {7: 1.0}
    l32 = {"edges": [_edge((), 7, root_p), _edge((7,), 1, {1: 1.0})]}
    l64 = {"edges": [_edge((), 7, root_p)]}                 # shallower line
    m = merge_reference_lines(l32, l64)
    reply = m["agreement"]["reply"]
    assert reply["state"] == "single_line"
    assert reply["in_3200"] is True and reply["in_6400"] is False
    assert m["agreement"]["root"]["state"] == "agree"
    # Neither line reached two ply at all.
    assert m["agreement"]["two_ply"]["state"] == "absent_both"


def test_priors_equality_is_asserted_where_both_rungs_captured_a_parent():
    """add_noise=False, so priors cannot change between rungs. The assertion is
    keyed on the PARENT PATH, not the edge -- two different edges can share a
    parent, which is exactly the case worth checking."""
    l32 = {"edges": [_edge((), 7, {7: 0.6, 8: 0.4})]}
    same_parent = {"edges": [_edge((), 8, {7: 0.6, 8: 0.4})]}
    merge_reference_lines(l32, same_parent)                 # different move, OK
    drifted = {"edges": [_edge((), 8, {7: 0.9, 8: 0.1})]}
    with pytest.raises(ValueError, match="priors"):
        merge_reference_lines(l32, drifted)


def test_the_ladder_freezes_both_deep_lines_and_both_parent_visit_maps():
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
    assert set(snaps["reference_lines"]) == {"at_3200", "at_6400", "merged"}
    assert set(snaps["parent_visits"]) == {"at_boundary", "at_400"}

    # Each map is anchored to its own instant: the root entry must equal the
    # tree capture taken at the same moment.
    pv_b, pv_4 = snaps["parent_visits"]["at_boundary"], snaps["parent_visits"]["at_400"]
    assert pv_b[()] == snaps["captures"]["at_boundary"]["root_visits"]
    assert pv_4[()] == snaps["captures"]["at_400"]["root_visits"]
    assert pv_4[()] >= pv_b[()]        # equal only if the boundary was the tail flush
    # ...and neither moved when legs 2-4 ran. The post-ladder tree is strictly
    # larger, so `>` here cannot be satisfied by two reads at the end.
    assert capture_parent_visits(pre.root)[()] > pv_4[()]

    # Every required edge carries the priors its rank will be computed from.
    assert all(e["parent_priors"]
               for e in snaps["reference_lines"]["merged"]["required_edges"])
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


def capture_parent_visits(root: MCTSNode, max_depth: int = 2
                          ) -> Dict[Tuple[int, ...], int]:
    """Visit counts for EVERY existing path through depth `max_depth`, at ONE
    instant (amendment 4).

    A complete map rather than a lookup of the reference line's paths, because
    the union of required edges is not known until leg 4 while this is captured
    during leg 1. A path ABSENT from the map has zero visits, where
    K(0) = min(n_legal, max(1, 0)) = 1 admits only rank 1.

    Bounded by the nodes that exist, not by the branching factor: at most
    `I + N_actual + 1` paths can exist at this instant, so this is hundreds of
    entries -- not the 250k a 500-wide two-ply enumeration would suggest.

    Keys are TUPLES, the natural type here. `build_atlas_corpus._jsonable`
    normalizes them at the artifact boundary; the pure modules keep tuples.
    """
    out: Dict[Tuple[int, ...], int] = {}
    frontier: List[Tuple[MCTSNode, Tuple[int, ...]]] = [(root, ())]
    while frontier:
        node, path = frontier.pop()
        out[path] = node.visit_count
        if len(path) < max_depth:
            frontier.extend((c, path + (mv,)) for mv, c in node.children.items())
    return out


def deep_reference_line(root: MCTSNode) -> Dict[str, Any]:
    """The two-ply reference line at ONE instant (amendment 4).

    Called at the end of leg 3 and again at the end of leg 4: the ladder is
    additive on ONE tree, so after leg 4 the 3,200 state exists nowhere and a
    single post-ladder summary could only ever describe 6,400.

    Emits EDGES -- `(parent_path, move)` plus the parent's priors, which is what
    ranks are computed from. It deliberately does NOT emit the parent's visit
    count: that is a deep-rung number, and retention must key on the boundary and
    B=400 maps instead. Not producing it is what makes the revision-1 defect
    unreachable rather than merely discouraged.
    """
    edges: List[Dict[str, Any]] = []
    node: Optional[MCTSNode] = root
    path: Tuple[int, ...] = ()
    for _ in range(3):                       # root move, reply, two-ply
        if node is None:
            break
        mv = visit_leader_move(node)         # canonical: ties by lowest move id
        if mv is None:
            break                            # nothing below here has a visit
        edges.append({
            "parent_path": path, "move": mv, "depth": len(path),
            "parent_priors": dict(node.priors or {}),
        })
        path = path + (mv,)
        node = node.children.get(mv)
    return {"edges": edges, "moves": [e["move"] for e in edges]}


DEPTH_NAMES = ("root", "reply", "two_ply")


def merge_reference_lines(line_3200: Dict[str, Any], line_6400: Dict[str, Any]
                          ) -> Dict[str, Any]:
    """The DEDUPLICATED UNION of both deep lines' edges (amendment 4).

    Agreements collapse to one edge; disagreements retain BOTH. Neither rung is
    declared truth and neither is called stable -- the point of the 3,200/6,400
    pair is that agreement is a FINDING, not an assumption.
    """
    by_key: Dict[Tuple[Tuple[int, ...], int], Dict[str, Any]] = {}
    priors_by_parent: Dict[Tuple[int, ...], Dict[int, float]] = {}
    at_depth: Dict[int, Dict[int, Tuple[Tuple[int, ...], int]]] = {}

    for rung, line in ((3200, line_3200), (6400, line_6400)):
        for e in line["edges"]:
            path = e["parent_path"]
            # Priors cannot change between rungs under the frozen
            # add_noise=False ladder, so ASSERT rather than assume. Keyed on the
            # PARENT, not the edge: two different edges can share a parent, and
            # that is exactly the case worth checking.
            prev = priors_by_parent.get(path)
            if prev is not None and prev != e["parent_priors"]:
                raise ValueError(
                    f"parent priors differ between deep rungs at path {path}; "
                    f"under the frozen add_noise=False ladder they cannot")
            priors_by_parent[path] = e["parent_priors"]

            key = (path, e["move"])
            if key in by_key:
                by_key[key]["sources"] = by_key[key]["sources"] + (rung,)
            else:
                by_key[key] = {**e, "sources": (rung,)}
            at_depth.setdefault(e["depth"], {})[rung] = key

    agreement: Dict[str, Any] = {}
    for depth, name in enumerate(DEPTH_NAMES):
        seen = at_depth.get(depth, {})
        in32, in64 = 3200 in seen, 6400 in seen
        if in32 and in64:
            # Agreement is equality of the COMPLETE edge: the same move id
            # under a different parent is a different edge.
            state = "agree" if seen[3200] == seen[6400] else "disagree"
        elif in32 or in64:
            # Amendment 4's "present in only one line": counted separately,
            # outside the agreement denominator. There is nothing to compare.
            state = "single_line"
        else:
            # Neither line reached this depth. A DIFFERENT missingness state --
            # collapsing it into single_line would report a comparison that was
            # never even half-available. Also outside the denominator.
            state = "absent_both"
        agreement[name] = {"in_3200": in32, "in_6400": in64, "state": state}

    return {
        # Sorted by (parent_path, move) so the union is reproducible run to run.
        "required_edges": [by_key[k] for k in sorted(by_key)],
        "agreement": agreement,
    }


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

**Who freezes what, and when.** `BatchSafeBoundaryObserver.on_flush_complete`
already receives `root`, so the boundary instant is the observer's to freeze:
alongside the existing `tracer_snapshot_at_boundary` it also stores
`capture_tree_state(root)` and `capture_parent_visits(root)`. Nothing else can
reach that instant — by the time `run_additive_ladder` regains control, leg 1 has
already finished.

| instant | frozen there | by |
|---|---|---|
| before leg 1 | `captures["at_start"]` | ladder |
| boundary flush, inside leg 1 | `captures["at_boundary"]`, `parent_visits["at_boundary"]`, `at_boundary` tracer snapshot | observer |
| after leg 1 (`B = 400`) | `captures["at_400"]`, `parent_visits["at_400"]`, `at_400` tracer snapshot | ladder |
| after leg 3 (`B = 3,200`) | `reference_lines["at_3200"]` | ladder |
| after leg 4 (`B = 6,400`) | `reference_lines["at_6400"]`, then `merged` | ladder |

**Select the deep rungs by LEG INDEX (`leg_idx == 2` and `leg_idx == 3`), never by
`running_B`.** CPU tests pass `increments=(80, 80, 80, 80)`, where `running_B` reaches
320 and never 3,200 — a `running_B == 3200` test would silently capture no deep line at
all and every downstream retention number would be computed over an empty union, with
green tests throughout.

In `selection_tracer.py`, each cell gains `lagged_first_touch_outside_events`,
incremented in the same pass using `k_of_n(parent_completed_visits + BATCH_LAG, ...)`.
`BATCH_LAG = 14` is a module constant.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_producer_closure.py tests/test_warm_prefix_replay.py tests/test_selection_tracer.py -v -p no:cacheprovider`
Expected: PASS — 13 new, plus the existing 32 warm-replay and 18 tracer tests all still green (the ladder's return shape is extended additively, so none of them changes).

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

### Task 3: Read-out A — classifier, string labels, bars, deployability, dual pipeline

**Files:**
- Modify: `scripts/GPU/alphazero/atlas_readout_a.py`
- Test: `tests/test_atlas_readout_a.py`

**Interfaces:**
- Consumes: rows shaped `{"label": str, "features_at_boundary": {...}, "features_at_400": {...}}` — which is exactly the artifact row Task 6 builds, so no translation layer exists to drift. The single-set entry points keep a `feature_key` with a `"features"` default, so Task 3's own synthetic rows stay one-set.
- Produces: `LABEL_TO_Y`; `prepare_rows(rows, feature_key="features") -> dict`; `standardize(...)`; `fit_ridge_logistic(...)`; `auc(...)`; `bootstrap_auc_lower_bound(...)`; `evaluate_detector(discovery, validation, *, feature_key="features") -> dict`; `deployability(...)`.
- Produces, for §6a's dual pipeline: `FEATURE_SETS`; `AUTHORITATIVE_FEATURE_SET`; `INSUFFICIENCY_VERDICTS`; `evaluate_detector_both(discovery, validation) -> dict`.

> **`classify_row` returns STRINGS.** A detector expecting numeric `1`/`0` would count
> every real row as neither class and silently train on nothing. `prepare_rows` filters
> to the two eligible classes, maps them explicitly, **rejects** rows with a missing
> feature per §6a, and reports the rejection count — and capacity is rechecked **after**
> rejection, because rejection is what can push a split below its own gate.

#### The dual pipeline (§6a "Read-out A runs on both feature sets")

The **identical** pipeline runs on the boundary features and on the `B = 400` features.
**The boundary remains authoritative** — it is the only instant at which a controller
could still act.

```text
boundary INSUFFICIENT_*                                    -> that verdict, as itself
boundary PASS                                              -> PASS
boundary FAIL, B=400 PASS, boundary rejected NO rows       -> LATE_ONLY_SEPARATION
boundary FAIL, B=400 PASS, boundary rejected ANY row       -> FAIL + lateness_blocked_by
boundary FAIL, B=400 not PASS                              -> FAIL
```

`LATE_ONLY_SEPARATION` requires **boundary `FAIL` and `B = 400` `PASS` — both, exactly**.
It is §6's stated failure condition, not a success: the information exists but arrives
too late to allocate the remaining budget.

#### Resolving the missing-feature contradiction — the amendment wins

Amendment 4 lists three things that are **not** evidence of lateness: a boundary
`INSUFFICIENT_CLASSES`, an `INSUFFICIENT_DISCOVERY_CLASSES`, **or a missing-feature
rejection.** The third is not a verdict — it is a *condition* that may or may not produce
one. Revision 2 read the clause as naming three verdicts and therefore let a boundary
rejection sit inside a `LATE_ONLY_SEPARATION`, while separately declaring unequal
complete-case sets "reported, not gated". Those two positions contradict each other.

**Resolved in favour of the amendment's literal text: any missing-feature rejection in
the boundary feature set blocks `LATE_ONLY_SEPARATION`.** Two reasons, and neither is
convenience:

- **It targets the exact failure direction.** Rejected rows shrink the boundary sample,
  and a smaller sample is *more* likely to miss the AUC and bootstrap bars. A boundary
  rejection is precisely what can manufacture the `FAIL` half of a lateness finding.
- **The alternative amends a frozen clause.** Narrowing "a missing-feature rejection" to
  "an insufficiency caused by rejection" is permissible only as a written amendment
  preceding the work — and it would weaken a safety clause to fit an implementation.
  Reading it as written costs nothing and fails closed.

Blocking counts rejections in **either split** of the boundary set: discovery rejections
change the fitted model, validation rejections change what it is scored on. The result is
reported as the boundary's own `FAIL`, inside the frozen verdict vocabulary, with
`lateness_blocked_by` and the counts recording why the lateness reading was withheld —
"reported as itself" rather than promoted, and never silently.

**What stays reported-not-gated.** Everything else about unequal row sets. `prepare_rows`
returns `kept_indices`, and `row_overlap` carries `n_common` / `identical` **separately
for discovery and for validation** — revision 2 computed validation only, so two models
fitted on different discovery rows could be reported as identical row sets. No gate is
invented on top of that; the one place it matters for a verdict is handled above.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_atlas_readout_a.py
from scripts.GPU.alphazero.atlas_readout_a import (
    AUTHORITATIVE_FEATURE_SET, FEATURE_SETS, INSUFFICIENCY_VERDICTS, LABEL_TO_Y,
    auc, bootstrap_auc_lower_bound, deployability, evaluate_detector,
    evaluate_detector_both, fit_ridge_logistic, prepare_rows, standardize,
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


# -- section 6a: the IDENTICAL pipeline on BOTH feature sets ------------------

def _dual_rows(boundary_separates, four_hundred_separates,
               n_misleading=20, n_stable=60):
    """20 misleading + 60 stable-negative: the smallest set that satisfies the
    frozen capacity gate (>=20 / >=25) AND can clear the <=25% flag-rate bar,
    since 20/80 is exactly 0.25. At 45 rows a perfect detector would fail its
    own flag rate at 44%, and the test would pin the wrong thing.

    The class counts are parameters because the ceiling binds them: a perfect
    detector flags every positive, so `n_misleading / (n_misleading + n_stable)`
    must stay <= 0.25 for a PASS to be reachable at all. Adding positives to a
    fixture is therefore NOT a safe way to grow it.
    """
    rows = []
    for i in range(n_misleading + n_stable):
        label = "misleading" if i < n_misleading else "stable_negative"
        hi = 1.0 if label == "misleading" else 0.0
        rows.append({
            "label": label,
            "features_at_boundary": {k: (hi if boundary_separates else 0.5)
                                     for k in FEATURE_NAMES},
            "features_at_400": {k: (hi if four_hundred_separates else 0.5)
                                for k in FEATURE_NAMES},
        })
    return rows


def test_the_pipeline_runs_on_both_frozen_feature_sets():
    assert FEATURE_SETS == ("features_at_boundary", "features_at_400")
    assert AUTHORITATIVE_FEATURE_SET == "features_at_boundary"
    rows = _dual_rows(True, True)
    r = evaluate_detector_both(rows, rows, replicates=64)
    assert set(r["per_feature_set"]) == set(FEATURE_SETS)
    assert r["authoritative"] == AUTHORITATIVE_FEATURE_SET


def test_the_boundary_result_is_authoritative_when_it_passes():
    rows = _dual_rows(True, False)
    r = evaluate_detector_both(rows, rows, replicates=64)
    assert r["per_feature_set"]["features_at_boundary"]["verdict"] == "PASS"
    assert r["per_feature_set"]["features_at_400"]["verdict"] == "FAIL"
    assert r["verdict"] == "PASS"            # a failing B=400 cannot demote it


def test_LATE_ONLY_SEPARATION_requires_boundary_FAIL_and_400_PASS():
    """Both, exactly. It is section 6's stated FAILURE condition: the
    information exists but arrives too late to allocate the last 80 sims."""
    rows = _dual_rows(False, True)
    r = evaluate_detector_both(rows, rows, replicates=64)
    assert r["per_feature_set"]["features_at_boundary"]["verdict"] == "FAIL"
    assert r["per_feature_set"]["features_at_400"]["verdict"] == "PASS"
    assert r["verdict"] == "LATE_ONLY_SEPARATION"


def test_both_sets_failing_is_a_plain_FAIL():
    rows = _dual_rows(False, False)
    assert evaluate_detector_both(rows, rows, replicates=64)["verdict"] == "FAIL"


def test_a_boundary_insufficiency_is_reported_as_itself_not_as_lateness():
    """Absence of evidence is never evidence of timing: a B=400 PASS must not
    promote an insufficient boundary result to LATE_ONLY_SEPARATION."""
    rows = _dual_rows(False, True)
    rows[0]["features_at_boundary"]["leader_breadth"] = None   # 19 misleading
    r = evaluate_detector_both(rows, rows, replicates=64)
    assert r["per_feature_set"]["features_at_boundary"]["verdict"] in \
        INSUFFICIENCY_VERDICTS
    assert r["per_feature_set"]["features_at_400"]["verdict"] == "PASS"
    assert r["verdict"] == "INSUFFICIENT_CLASSES"


def test_a_boundary_REJECTION_blocks_LATE_ONLY_SEPARATION():
    """Amendment 4 lists a missing-feature rejection alongside the two
    insufficiency verdicts as something that cannot establish lateness.

    Rejections shrink the boundary sample, and a smaller sample is MORE likely
    to miss the AUC and bootstrap bars -- so a rejection is exactly what could
    manufacture the FAIL half of a lateness finding.

    Sizing is exact and both halves are load-bearing. 21/84 keeps the B=400
    flag rate at exactly the 0.25 ceiling, so a perfect detector still PASSES;
    rejecting one boundary misleading row leaves 20/83, which still clears the
    >=20 / >=25 capacity gate, so the boundary reaches the bars and returns a
    genuine FAIL rather than an INSUFFICIENT_CLASSES. Adding three MISLEADING
    rows instead would give 23/83 = 27.7% and the B=400 half could never pass.
    """
    rows = _dual_rows(False, True, n_misleading=21, n_stable=63)
    rows[0]["features_at_boundary"]["leader_breadth"] = None
    r = evaluate_detector_both(rows, rows, replicates=64)
    assert r["per_feature_set"]["features_at_boundary"]["verdict"] == "FAIL"
    assert r["per_feature_set"]["features_at_400"]["verdict"] == "PASS"
    # Reported as the boundary's OWN result, not promoted to a timing finding.
    assert r["verdict"] == "FAIL"
    assert r["lateness_blocked_by"] == "boundary_missing_feature_rejections"


def test_row_overlap_is_reported_for_DISCOVERY_and_validation_separately():
    """Missing discovery features make the two models train on different rows,
    which a validation-only overlap would report as identical row sets."""
    rows = _dual_rows(True, True)
    disc = [dict(r, features_at_400=dict(r["features_at_400"])) for r in rows]
    disc[0]["features_at_400"]["root_policy_entropy"] = None
    r = evaluate_detector_both(disc, rows, replicates=64)
    assert r["per_feature_set"]["features_at_400"]["rejected_missing_features"] == 0
    assert r["row_overlap"]["validation"]["identical"] is True
    assert r["row_overlap"]["discovery"]["identical"] is False
    assert r["row_overlap"]["discovery"]["n_common"] == 79
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


def prepare_rows(rows: Sequence[Dict[str, Any]],
                 feature_key: str = "features") -> Dict[str, Any]:
    """String labels -> numeric, ineligible classes dropped, missing-feature
    rows REJECTED and counted (section 6a).

    `feature_key` selects which frozen capture to read, so the SAME pipeline
    serves both feature sets. `kept_indices` is what lets the dual pipeline
    report whether the two sets ran on the same rows.
    """
    feats, y, kept = [], [], []
    dropped = rejected = 0
    for i, r in enumerate(rows):
        if r["label"] not in LABEL_TO_Y:
            dropped += 1                       # ambiguous / no_stable_reference
            continue
        # A MISSING capture rejects the row exactly like a missing feature:
        # `or {}` makes every feature None rather than raising a KeyError that a
        # caller might be tempted to catch and default.
        f = r.get(feature_key) or {}
        if any(f.get(k) is None for k in FEATURE_NAMES):
            rejected += 1
            continue
        feats.append(f)
        y.append(LABEL_TO_Y[r["label"]])
        kept.append(i)
    return {"features": feats, "y": y, "kept_indices": kept,
            "dropped_ineligible": dropped, "rejected_missing_features": rejected}


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
                      seed: int = BOOTSTRAP_SEED,
                      replicates: int = BOOTSTRAP_REPLICATES,
                      feature_key: str = "features") -> Dict[str, Any]:
    """Section 6's frozen bars. Fails CLOSED, and capacity is checked AFTER
    missing-feature rejection.

    `seed` and `replicates` default to the FROZEN values and are parameters only
    so CPU tests can run a cheap bootstrap; nothing on the measurement path
    passes either. `feature_key` is what makes this the same pipeline for both
    feature sets rather than a second implementation of it.
    """
    d = prepare_rows(discovery, feature_key)
    v = prepare_rows(validation, feature_key)
    v_pos, v_neg = v["y"].count(1), v["y"].count(0)
    base = {"feature_set": feature_key,
            "n_misleading": v_pos, "n_stable_negative": v_neg,
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
    lb = bootstrap_auc_lower_bound(sv, v["y"], seed=seed, replicates=replicates)
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


# -- section 6a: Read-out A runs on BOTH feature sets -------------------------

FEATURE_SETS = ("features_at_boundary", "features_at_400")
AUTHORITATIVE_FEATURE_SET = "features_at_boundary"
INSUFFICIENCY_VERDICTS = ("INSUFFICIENT_CLASSES",
                          "INSUFFICIENT_DISCOVERY_CLASSES")


def evaluate_detector_both(discovery: Sequence[Dict[str, Any]],
                           validation: Sequence[Dict[str, Any]],
                           seed: int = BOOTSTRAP_SEED,
                           replicates: int = BOOTSTRAP_REPLICATES
                           ) -> Dict[str, Any]:
    """The IDENTICAL pipeline on both frozen feature sets (amendment 6a).

    The boundary remains AUTHORITATIVE -- it is the only instant at which a
    controller could still act on the result.

        boundary PASS                    -> PASS
        boundary FAIL  and B=400 PASS    -> LATE_ONLY_SEPARATION
        boundary FAIL  and B=400 not PASS-> FAIL
        boundary INSUFFICIENT_*          -> that verdict, reported as itself

    LATE_ONLY_SEPARATION means the information exists but arrives too late to
    allocate the remaining budget -- section 6's stated FAILURE condition, not a
    success.
    """
    per = {fs: evaluate_detector(discovery, validation, seed=seed,
                                 replicates=replicates, feature_key=fs)
           for fs in FEATURE_SETS}
    auth = per[AUTHORITATIVE_FEATURE_SET]["verdict"]
    late = per["features_at_400"]["verdict"]

    # Amendment 4 lists a MISSING-FEATURE REJECTION alongside the two
    # insufficiency verdicts as something that cannot establish lateness.
    # Rejections shrink the boundary sample and a smaller sample is MORE likely
    # to miss the bars, so a rejection is exactly what could manufacture the
    # FAIL half of a lateness finding. Counted in BOTH splits: discovery
    # rejections change the fitted model, validation rejections change what it
    # is scored on.
    boundary_rejections = sum(
        prepare_rows(split, feature_key=AUTHORITATIVE_FEATURE_SET
                     )["rejected_missing_features"]
        for split in (discovery, validation))

    blocked = None
    if auth in INSUFFICIENCY_VERDICTS:
        # An ABSENCE of evidence, never evidence about timing.
        verdict = auth
    elif auth == "PASS":
        verdict = "PASS"
    elif late == "PASS" and boundary_rejections == 0:
        verdict = "LATE_ONLY_SEPARATION"          # boundary FAIL and 400 PASS
    elif late == "PASS":
        # Reported as the boundary's OWN result, inside the frozen verdict
        # vocabulary, with the reason recorded rather than silently dropped.
        verdict, blocked = "FAIL", "boundary_missing_feature_rejections"
    else:
        verdict = "FAIL"

    def overlap(split):
        kept = {fs: set(prepare_rows(split, feature_key=fs)["kept_indices"])
                for fs in FEATURE_SETS}
        a, b = kept[FEATURE_SETS[0]], kept[FEATURE_SETS[1]]
        return {"n_common": len(a & b), "identical": a == b}

    return {
        "verdict": verdict,
        "authoritative": AUTHORITATIVE_FEATURE_SET,
        "per_feature_set": per,
        "boundary_rejections": boundary_rejections,
        "lateness_blocked_by": blocked,
        # REPORTED, never gated. BOTH splits: a validation-only overlap would
        # call two models identical when they were fitted on different rows.
        "row_overlap": {"discovery": overlap(discovery),
                        "validation": overlap(validation)},
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_a.py -v -p no:cacheprovider`
Expected: PASS — 27 passed (7 from Task 2, 13 here, 7 dual-pipeline).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_readout_a.py tests/test_atlas_readout_a.py
git commit -m "feat(atlas-s4): Read-out A frozen bars, deployability, dual-feature-set pipeline"
```

---

### Task 4: Read-out B — four rungs, natural convergence, strata

**Files:**
- Create: `scripts/GPU/alphazero/atlas_readout_b.py`
- Test: `tests/test_atlas_readout_b.py`

**Interfaces:**
- Consumes: a **calibration row**, a plain dict — `{"legs": [...], "phase": str, "flat_policy": bool, "near_even": bool}`. A bare `list[LegResult]` cannot identify the strata §7 requires. There is no row *class*: revision 1's interface block named a `CalibrationRow` that nothing defines, which is the declared-but-absent trap this plan has now hit four times.
- Produces: `COLLAPSE_TOP_SHARE`; `MIN_ELIGIBLE_TRIGGERS`; `MIN_CONVERGENT_RATE`; `BASE_RATE_MARGIN`; `GATE_NAMES`; `gate_triggers(legs, hi=1600)`; `closes_half(...)`; `convergent(legs, ref)`; `compound_narrowing(rows, hi=1600) -> Optional[bool]`; `calibrate_gate(rows, gate_name)`; `natural_convergence_report(rows)`; `by_stratum_summary(rows, gate_name)`.

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
Expected: PASS — 14 passed.

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
- Consumes: a **Read-out C row** — `{"snapshots": <run_additive_ladder's snapshots dict, verbatim>, "label": str, "phase": str, "flat_policy": bool, "near_even": bool}`. The whole producer document travels as **one** key rather than being re-assembled from three, because a consumer that reconstructs its producer's document is the seam that cost v18 four contract defects.
- Produces: `STRATA`; `INSTANTS`; `GATING_INSTANT`; `REQUIRED_RATES`; `STABLE_REFERENCE_LABELS`; `static_retention(...)`; `edge_retention(edge, parent_visits, shape)`; `intervention_from_snapshots(snapshots, shape_key, *, instant)`; `classify_strata(row)`; **`classify_edge_strata(edge)`**; `aggregate_shape(rows, shape)`; `validation_verdict(aggregate)`; `select_shape(per_shape)`; `select_on_discovery_validate_on_selected(discovery, validation)`.
- **The row is exactly what `atlas_artifact.build_row` returns.** Task 6 stores the Stage 3 producer document once, under `snapshots`, so Read-out C consumes an artifact row directly and no surrogate translation exists between them.

> **`K(n)` uses EFFECTIVE parent visits, read from the instant's own map** (§6a).
> At the warm root that is `I + N_actual` at the boundary and `I + 400` after leg 1 —
> **not** the nominal 320, and **not** the `I + 6400` of the final tree. Using a nominal
> or a final-tree count would narrow or widen the admitted set, understating retention
> and overstating intervention, in the same direction as the batch lag and on top of it.

#### What amendment 4 changes here

| | revision 1 | now |
|---|---|---|
| required moves | one line's root move, its reply, its two-ply move | the **deduplicated union** of `(parent path, move)` edges from **both** deep lines; both replies retained when they differ |
| `K(n)`'s `n` | `root_effective_visits` off the final 6,400 tree | the **instant's** parent-visit map — absent path ⇒ **0** visits ⇒ `K(0) = 1` |
| where retention is judged | once | at the **boundary and at `B = 400`**; root and reply floors must pass at **both** |
| what drives the bars | the boundary snapshot | the **`B = 400`** intervention; the boundary one is reported |
| flat-policy strata | row-level | **edge-level**, via `classify_edge_strata` — a row can hold both flat and non-flat reference parents |
| who feeds the retention bars | every row | **stable-reference-eligible rows only** — the floors are about *stable* deep moves, and a `no_stable_reference` row has none. Selection-event counters still cover every row |
| the aggregate | four rates | plus depth buckets, event-weighted excluded prior mass, forced-root bypasses, forced-simulation counts, per-stratum retention, `retention_rows`, and the agreement states |
| the validation aggregate | computed | **judged**, by `validation_verdict` |

**Hoisting rule, so `select_shape` stays a flat read.** Because the floors must hold at
both instants, the hoisted `root_retention` / `depth1_retention` / `descendant_retention`
are the **worse of the two instants** — `min(a, b) >= bar` is exactly "passes at both",
and `None` if either is undefined, since an undefined rate is not a satisfied bar. The
hoisted `misleading_intervention` / `stable_intervention` are the **`B = 400`** numbers.
Per-instant values stay in `instants` for reporting, so nothing is hidden by the hoist.

**Pooling rule.** Cohort counters sum numerators and denominators; they are never a mean
of per-row rates. Event-weighted excluded prior mass is `Σ mass / Σ eligible_events`, so
a 3-event row does not weigh the same as a 400-event one.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_readout_c.py
import pytest

from scripts.GPU.alphazero.atlas_readout_c import (
    GATING_INSTANT, INSTANTS, MISLEADING_INTERVENTION_BAR, REQUIRED_RATES,
    RETENTION_DEPTH1_BAR, RETENTION_ROOT_BAR, STABLE_INTERVENTION_CEILING,
    STABLE_REFERENCE_LABELS, STRATA, aggregate_shape, classify_edge_strata,
    classify_strata, edge_retention, intervention_from_snapshots,
    select_on_discovery_validate_on_selected, select_shape, static_retention,
    validation_verdict,
)
from scripts.GPU.alphazero.selection_tracer import WIDENING_SHAPES

SHAPE = ("c4a05", 4.0, 0.5)


def _priors(n, best=0):
    """Descending, so prior rank == move id + 1."""
    return {i: (1.0 if i == best else 0.5 - i * 1e-4) for i in range(n)}


def _edge(path, move, priors=None, sources=(3200, 6400)):
    return {"parent_path": path, "move": move, "depth": len(path),
            "parent_priors": priors if priors is not None else _priors(500),
            "sources": sources}


def _merged(edges=None, agreement=None):
    """`merge_reference_lines` output: the union plus the agreement report."""
    return {
        "required_edges": edges if edges is not None else [
            _edge((), 0), _edge((0,), 1, _priors(400))],
        "agreement": agreement or {
            "root": {"in_3200": True, "in_6400": True, "state": "agree"},
            "reply": {"in_3200": True, "in_6400": True, "state": "agree"},
            "two_ply": {"in_3200": False, "in_6400": False,
                        "state": "absent_both"}},
    }


def _cell(elig=200, outside=30, ft=100, ft_out=15, lagged=12, mass=80.0):
    """One tracer cell, shaped exactly like SelectionTracer.snapshot emits."""
    return {"eligible_events": elig, "outside_events": outside,
            "first_touch_events": ft, "first_touch_outside_events": ft_out,
            "lagged_first_touch_outside_events": lagged,
            "excluded_prior_mass": mass,
            "outside_rate": (outside / elig) if elig else None,
            "first_touch_outside_rate": (ft_out / ft) if ft else None,
            "mean_excluded_prior_mass": (mass / elig) if elig else None}


def _tracer(overall=None, within_forced=5, bypass=2, bypass_out=1):
    o = overall if overall is not None else _cell()
    block = {**{k: dict(o) for k in ("overall", "0", "1", "2+")},
             "forced_root_bypass_events": bypass,
             "forced_root_bypass_outside_events": bypass_out,
             "forced_root_bypass_outside_rate": ((bypass_out / bypass)
                                                 if bypass else None),
             "meaningfully_affected": (o["first_touch_outside_rate"] is not None
                                       and o["first_touch_outside_rate"] >= 0.10)}
    return {"by_shape": {n: dict(block) for n, _c, _a in WIDENING_SHAPES},
            "within_forced_events": within_forced}


def _pv(root=463, reply=90):
    return {(): root, (0,): reply}


def _snaps(boundary=None, at400=None, merged=None, pv=None):
    """The ladder's snapshots dict, verbatim -- one producer document."""
    return {"at_boundary": _tracer() if boundary is None else boundary,
            "at_400": _tracer() if at400 is None else at400,
            "reference_lines": {"at_3200": None, "at_6400": None,
                                "merged": merged or _merged()},
            "parent_visits": {"at_boundary": pv or _pv(),
                              "at_400": pv or _pv()}}


def _row(label="misleading", phase="late", flat=False, near_even=False,
         snaps=None):
    return {"snapshots": snaps or _snaps(), "label": label, "phase": phase,
            "flat_policy": flat, "near_even": near_even}


def test_frozen_bars_and_strata_are_pinned():
    assert RETENTION_ROOT_BAR == 0.95 and RETENTION_DEPTH1_BAR == 0.90
    assert MISLEADING_INTERVENTION_BAR == 0.50
    assert STABLE_INTERVENTION_CEILING == 0.25
    assert set(STRATA) == {"late", "near_even", "root_flat",
                           "locally_flat_depth1", "locally_flat_depth2"}
    assert INSTANTS == ("at_boundary", "at_400")
    assert GATING_INSTANT == "at_400"           # amendment 4: B=400 drives bars
    # An ALLOW-list: a label added later must not be admitted by default.
    assert set(STABLE_REFERENCE_LABELS) == {"misleading", "stable_negative",
                                            "ambiguous"}


def test_static_retention_uses_EFFECTIVE_parent_visits():
    """K(n) keys on completed visits, which at a warm root include I."""
    wide = static_retention(_priors(500), [80], n_at_selection=463, shape=SHAPE)
    narrow = static_retention(_priors(500), [80], n_at_selection=320, shape=SHAPE)
    assert wide["k"] > narrow["k"]
    assert wide["retained"] == 1 and narrow["retained"] == 0


def test_static_retention_of_nothing_is_None():
    assert static_retention(_priors(10), [], 400, SHAPE)["rate"] is None


def test_edge_retention_reads_the_INSTANT_parent_visit_map():
    """n comes from THAT instant's map, never a nominal budget and never the
    6,400 tree. An ABSENT path has zero visits, where K(0) = 1 admits rank 1
    only."""
    edge = _edge((), 80)                              # prior rank 81
    wide = edge_retention(edge, {(): 463}, SHAPE)     # K = 87
    narrow = edge_retention(edge, {(): 320}, SHAPE)   # K = 72
    assert wide["k"] > narrow["k"]
    assert wide["retained"] is True and narrow["retained"] is False
    absent = edge_retention(edge, {}, SHAPE)
    assert absent["n"] == 0 and absent["k"] == 1 and absent["retained"] is False
    assert edge_retention(_edge((), 0), {}, SHAPE)["retained"] is True


def test_retention_covers_the_deduplicated_union_of_edges():
    """When the deep lines disagree BOTH replies are required, so both count
    toward the depth-1 denominator. Neither is truth."""
    merged = _merged([_edge((), 0),
                      _edge((0,), 1, _priors(400), sources=(3200,)),
                      _edge((0,), 300, _priors(400), sources=(6400,))])
    a = aggregate_shape([_row(snaps=_snaps(merged=merged))], SHAPE)
    reply = a["instants"]["at_400"]["by_role"]["reply"]
    assert reply["required"] == 2                 # both replies retained
    assert reply["retained"] == 1                 # rank 301 is far outside K(90)


def test_a_retention_floor_must_pass_at_BOTH_instants():
    """Amendment 4. The hoisted number is the WORSE instant, so `>= bar` is
    exactly "passed at both"."""
    snaps = _snaps(merged=_merged([_edge((), 80)]))          # rank 81
    snaps["parent_visits"] = {"at_boundary": {(): 320},      # K = 72 -> missed
                              "at_400": {(): 463}}           # K = 87 -> retained
    a = aggregate_shape([_row(snaps=snaps)], SHAPE)
    assert a["instants"]["at_400"]["root_retention"] == 1.0
    assert a["instants"]["at_boundary"]["root_retention"] == 0.0
    assert a["root_retention"] == 0.0            # the worse one, not the better


def test_the_bars_use_the_B400_intervention_and_report_the_boundary_one():
    snaps = _snaps(boundary=_tracer(_cell(ft=100, ft_out=2, lagged=1)),
                   at400=_tracer(_cell(ft=100, ft_out=40, lagged=35)))
    a = aggregate_shape([_row(label="misleading", snaps=snaps)], SHAPE)
    assert a["gated_on"] == "at_400"
    assert a["instants"]["at_400"]["misleading_intervention"] == 1.0
    assert a["instants"]["at_boundary"]["misleading_intervention"] == 0.0
    assert a["misleading_intervention"] == 1.0          # the B=400 number


def test_a_missing_snapshot_is_NO_SNAPSHOT_not_zero():
    """A row whose boundary never fired has no snapshot. That is not an
    intervention rate of zero."""
    snaps = _snaps()
    snaps["at_boundary"] = None
    r = intervention_from_snapshots(snaps, "c4a05", instant="at_boundary")
    assert r["verdict"] == "NO_SNAPSHOT" and r["meaningfully_affected"] is None


def test_intervention_requires_the_PRODUCED_lagged_bound():
    snaps = _snaps(at400=_tracer(_cell(ft=100, ft_out=12, lagged=8)))
    r = intervention_from_snapshots(snaps, "c4a05", instant="at_400")
    assert r["meaningfully_affected"] is None      # None, not False
    assert r["verdict"] == "INCONCLUSIVE"


def test_intervention_passes_when_both_bounds_clear():
    snaps = _snaps(at400=_tracer(_cell(ft=100, ft_out=15, lagged=12)))
    r = intervention_from_snapshots(snaps, "c4a05", instant="at_400")
    assert r["meaningfully_affected"] is True and r["verdict"] == "OK"


def test_classify_strata_reads_the_row_not_a_bare_leg_list():
    s = classify_strata(_row(phase="late", flat=True, near_even=True))
    assert {"late", "root_flat", "near_even"} <= s


def test_local_flat_strata_are_EDGE_level_not_row_level():
    """A row can hold both flat and non-flat reference parents; pooling them
    would hide the contrast the stratum exists to expose."""
    flat_priors = {i: 1.0 / 500 for i in range(500)}
    assert "locally_flat_depth1" in classify_edge_strata(
        {"depth": 1, "parent_priors": flat_priors})
    assert "locally_flat_depth2" in classify_edge_strata(
        {"depth": 2, "parent_priors": flat_priors})
    assert classify_edge_strata(
        {"depth": 1, "parent_priors": {0: 0.9, 1: 0.05, 2: 0.05}}) == set()


def test_a_flat_ROOT_edge_gets_no_local_stratum():
    """Depth 0 is the root, not a local parent. An `else depth2` fallthrough
    would invent a stratum membership the edge does not have."""
    flat_priors = {i: 1.0 / 500 for i in range(500)}
    assert classify_edge_strata({"depth": 0, "parent_priors": flat_priors}) == set()
    assert classify_edge_strata({"parent_priors": flat_priors}) == set()
    assert classify_edge_strata({"depth": 7, "parent_priors": flat_priors}) == set()


def test_per_stratum_retention_uses_edge_level_flatness():
    """One row, one flat reference parent and one concentrated one. `_priors`
    is itself flat under the frozen definition -- normalized entropy ~1.0 and a
    top prior of ~0.005 -- so the non-flat case must be built explicitly."""
    flat = {i: 1.0 / 500 for i in range(500)}
    sharp = {0: 0.5, 1: 0.3, 2: 0.2}                      # NOT flat
    merged = _merged([_edge((), 0),
                      _edge((0,), 1, flat),               # locally flat, depth 1
                      _edge((0, 1), 2, sharp)])
    snaps = _snaps(merged=merged, pv={(): 463, (0,): 90, (0, 1): 20})
    a = aggregate_shape([_row(phase="late", snaps=snaps)], SHAPE)
    st = a["instants"]["at_400"]["by_stratum"]
    assert st["locally_flat_depth1"]["required"] == 1
    assert st["locally_flat_depth2"]["required"] == 0
    assert st["locally_flat_depth2"]["rate"] is None      # None, never 0.0
    assert st["late"]["required"] == 3                    # row-level: all edges


def test_aggregate_excludes_INCONCLUSIVE_rows_from_the_denominator():
    """Folding them in as either outcome would invent a measurement."""
    rows = [_row(snaps=_snaps(at400=_tracer(_cell(ft=100, ft_out=15, lagged=12)))),
            _row(snaps=_snaps(at400=_tracer(_cell(ft=100, ft_out=12, lagged=8))))]
    a = aggregate_shape(rows, SHAPE)
    assert a["misleading_denominator"] == 1
    assert a["inconclusive"] == 1


def test_aggregate_rate_is_None_when_the_denominator_empties():
    rows = [_row(snaps=_snaps(at400=_tracer(_cell(ft=100, ft_out=12, lagged=8))))]
    a = aggregate_shape(rows, SHAPE)
    assert a["misleading_intervention"] is None


def test_aggregate_reports_depth_buckets_forced_counts_and_agreement():
    """Section 8's online aggregates, pooled across the cohort."""
    a = aggregate_shape([_row(), _row()], SHAPE)
    c = a["counters"]["at_400"]
    assert set(c["by_depth"]) == {"0", "1", "2+"}
    assert c["eligible_events"] == 400                    # 2 rows x 200
    # Forced-root bypasses are reported SEPARATELY, never in the primary
    # intervention denominator.
    assert c["forced_root_bypass_events"] == 4
    assert c["forced_root_bypass_outside_rate"] == 0.5
    assert c["within_forced_events"] == 10
    # Agreement is reported and adds NO gate. The two missingness states are
    # counted separately, and neither is in the denominator.
    assert a["agreement"]["reply"]["agree_rate"] == 1.0
    assert a["agreement"]["two_ply"]["absent_both"] == 2
    assert a["agreement"]["two_ply"]["single_line"] == 0
    assert a["agreement"]["two_ply"]["agree_rate"] is None


def test_retention_bars_exclude_rows_without_a_stable_reference():
    """Section 8's floors are about STABLE deep moves. A row whose 3,200 and
    6,400 rungs never agreed has none, so it contributes no required edges --
    but its selection events still count, because those describe what widening
    would have done regardless of the label."""
    rows = [_row(label="misleading"), _row(label="no_stable_reference")]
    a = aggregate_shape(rows, SHAPE)
    at400 = a["instants"]["at_400"]
    assert at400["retention_rows"] == 1                 # not 2
    assert at400["by_role"]["root"]["required"] == 1    # one row's edges only
    assert a["rows_without_stable_reference"] == 1
    # Event counters cover EVERY row.
    assert a["counters"]["at_400"]["eligible_events"] == 400
    # ...and the excluded row is in neither intervention denominator.
    assert at400["misleading_denominator"] == 1
    assert at400["stable_denominator"] == 0


def test_excluded_prior_mass_pools_event_wise_across_rows():
    """Sum the mass and sum the events. A mean of per-row means would weight a
    10-event row the same as a 990-event one."""
    small = _snaps(at400=_tracer(_cell(elig=10, mass=1.0)))       # row mean 0.10
    large = _snaps(at400=_tracer(_cell(elig=990, mass=495.0)))    # row mean 0.50
    a = aggregate_shape([_row(snaps=small), _row(snaps=large)], SHAPE)
    # Pooled 496/1000; a mean of per-row means would have given 0.30.
    assert a["counters"]["at_400"]["mean_excluded_prior_mass"] == pytest.approx(0.496)


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


# -- amendment 6a: the validation aggregate is JUDGED, not merely computed ----

def test_validation_verdict_precedence_is_FAIL_then_INCONCLUSIVE_then_PASS():
    """A DEFINED miss is evidence and outranks a gap in the evidence."""
    good = {"root_retention": 0.99, "depth1_retention": 0.95,
            "misleading_intervention": 0.60, "stable_intervention": 0.10}
    assert validation_verdict(good)["verdict"] == "PASS"
    assert validation_verdict(
        dict(good, misleading_intervention=None))["verdict"] == "INCONCLUSIVE"
    assert validation_verdict(dict(good, root_retention=0.50))["verdict"] == "FAIL"

    # BOTH at once -- the case the precedence exists for. Without an ordering
    # this result satisfies two verdicts simultaneously.
    r = validation_verdict(dict(good, root_retention=0.50,
                                misleading_intervention=None))
    assert r["verdict"] == "FAIL"
    assert r["failed"] == ["root_retention"]
    assert r["undefined"] == ["misleading_intervention"]


def test_the_ceiling_is_judged_as_a_ceiling_not_a_floor():
    good = {"root_retention": 0.99, "depth1_retention": 0.95,
            "misleading_intervention": 0.60, "stable_intervention": 0.10}
    assert validation_verdict(dict(good, stable_intervention=0.90))["verdict"] == "FAIL"


def test_the_tie_break_retention_is_not_a_required_rate():
    """descendant_retention breaks exact ties in shape selection; it is not a
    bar, so an undefined one must not turn a passing aggregate INCONCLUSIVE."""
    assert "descendant_retention" not in REQUIRED_RATES
    a = {"root_retention": 0.99, "depth1_retention": 0.95,
         "misleading_intervention": 0.60, "stable_intervention": 0.10,
         "descendant_retention": None}
    assert validation_verdict(a)["verdict"] == "PASS"


def _cohort():
    """Misleading rows that widening would intervene on, stable-negative rows
    it would leave alone -- the shape a passing feasibility result has."""
    hot = _tracer(_cell(ft=100, ft_out=40, lagged=35))
    cold = _tracer(_cell(ft=100, ft_out=2, lagged=1))
    return ([_row(label="misleading", snaps=_snaps(boundary=hot, at400=hot))
             for _ in range(2)]
            + [_row(label="stable_negative", snaps=_snaps(boundary=cold,
                                                          at400=cold))
               for _ in range(2)])


def test_the_selected_shape_receives_the_three_way_verdict():
    rows = _cohort()
    r = select_on_discovery_validate_on_selected(rows, rows)
    assert r["selected"] is not None
    assert set(r["validated"]) == {r["selected"]}
    assert r["validation_verdict"]["verdict"] == "PASS"


def test_no_selected_shape_means_no_verdict_to_give():
    """NO_SHAPE_PASSES is not an INCONCLUSIVE validation -- nothing was
    validated, so there is no validation aggregate to judge."""
    r = select_on_discovery_validate_on_selected([_row()], [_row()])
    assert r["selected"] is None and r["verdict"] == "NO_SHAPE_PASSES"
    assert r["validation_verdict"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_c.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/GPU/alphazero/atlas_readout_c.py
"""Atlas Read-out C -- design section 8 and amendments 6a / 4, FROZEN.

Counterfactual COVERAGE analysis. It cannot prove progressive widening would
improve search, because applying widening changes the later tree.

Everything here is evaluated at TWO instants -- the batch-safe boundary and
nominal B = 400 -- because those are the horizons a widening rule would actually
see. The 6,400 tree is never a retention horizon, and no function here can reach
one: the producer does not emit deep-rung visit counts at all.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .selection_tracer import (
    DEPTH_BUCKETS, MEANINGFUL_INTERVENTION_FIRST_TOUCH_RATE, WIDENING_SHAPES,
    k_of_n,
)
# DEPTH_NAMES ("root", "reply", "two_ply") is the reference line's own vocabulary
# and belongs with the producer that emits it. Imported rather than restated:
# a second copy of a frozen constant is how the two drift apart.
from .warm_prefix_replay import DEPTH_NAMES

RETENTION_ROOT_BAR = 0.95
RETENTION_DEPTH1_BAR = 0.90
MISLEADING_INTERVENTION_BAR = 0.50
STABLE_INTERVENTION_CEILING = 0.25
FLAT_ENTROPY_BAR = 0.90
FLAT_TOP_PRIOR_BAR = 0.025

STRATA = ("late", "near_even", "root_flat",
          "locally_flat_depth1", "locally_flat_depth2")

# Amendment 4: retention is judged at BOTH instants, and the B=400 intervention
# drives the feasibility bars while the boundary one is reported.
INSTANTS = ("at_boundary", "at_400")
GATING_INSTANT = "at_400"

# Section 8's floors are about "stable deep root moves" and "stable depth-1
# replies", so only rows that HAVE a stable deep reference may contribute to
# them. A `no_stable_reference` row's two deep rungs disagree; its reference
# line is not a stable deep move and cannot be evidence of retaining one.
#
# An explicit ALLOW-list, never `label != "no_stable_reference"`: a label added
# later would silently be admitted by the deny-list form.
STABLE_REFERENCE_LABELS = ("misleading", "stable_negative", "ambiguous")

# The rates the validation verdict requires. `descendant_retention` is NOT one:
# it breaks exact ties in shape selection and is not a bar, so an undefined one
# must not turn a passing aggregate INCONCLUSIVE.
REQUIRED_RATES = ("root_retention", "depth1_retention",
                  "misleading_intervention", "stable_intervention")
_BARS = {"root_retention": (RETENTION_ROOT_BAR, "floor"),
         "depth1_retention": (RETENTION_DEPTH1_BAR, "floor"),
         "misleading_intervention": (MISLEADING_INTERVENTION_BAR, "floor"),
         "stable_intervention": (STABLE_INTERVENTION_CEILING, "ceiling")}


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


def edge_retention(edge: Dict[str, Any],
                   parent_visits: Dict[Tuple[int, ...], int],
                   shape: Tuple[str, float, float]) -> Dict[str, Any]:
    """Would this required edge have been admitted AT THIS INSTANT?

    `n` is the parent's effective completed visit count read from the instant's
    OWN map -- `I + N_actual` at the boundary, `I + 400` after leg 1; never the
    nominal 320 and never the final 6,400 tree (amendments 6a and 4).

    A path ABSENT from the map has ZERO visits. `K(0) = min(n_legal, max(1, 0))
    = 1` through the `max(1, ...)` floor, so rank 1 is still admitted there --
    a real admission, not a vacuous one.
    """
    n = parent_visits.get(edge["parent_path"], 0)
    r = static_retention(edge["parent_priors"], [edge["move"]], n, shape)
    return {"retained": r["retained"] == 1, "k": r["k"], "n": n,
            "depth": edge["depth"]}


def intervention_from_snapshots(snapshots: Dict[str, Any], shape_key: str,
                                *, instant: str) -> Dict[str, Any]:
    """Meaningful intervention with the DIRECTIONAL lag bound, at ONE instant.

    The lag is conservative for retention and ANTI-conservative for
    intervention, so the threshold must also pass under K(n+14) -- a counter the
    tracer PRODUCES, never a caller-supplied number. Passing only under K(n) is
    INCONCLUSIVE, not a pass.

    `instant` is keyword-only with NO default. Amendment 4 gates on B=400 and
    reports the boundary separately, and a defaulted instant is exactly how a
    caller silently gets the other one.
    """
    snap = snapshots.get(instant)
    if snap is None:
        # A row whose boundary never fired has no snapshot at that instant.
        # That is not a rate of zero and not an intervention of False.
        return {"first_touch_outside_rate": None, "lagged_rate": None,
                "meaningfully_affected": None, "verdict": "NO_SNAPSHOT"}
    cell = snap["by_shape"][shape_key]["overall"]
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
    if not priors or not _is_flat(priors):
        return s
    # EXPLICIT branches. An `else depth2` fallthrough would map a flat ROOT edge
    # (depth 0) -- and any malformed or missing depth -- to locally_flat_depth2,
    # inventing a stratum membership the edge does not have.
    depth = edge.get("depth")
    if depth == 1:
        s.add("locally_flat_depth1")
    elif depth == 2:
        s.add("locally_flat_depth2")
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


def _rate(num: float, den: float) -> Optional[float]:
    """Zero denominator -> None. Never 0.0, never False."""
    return (num / den) if den else None


def _pair() -> Dict[str, int]:
    return {"retained": 0, "required": 0}


def _empty_counters() -> Dict[str, Any]:
    def z():
        return {"eligible_events": 0, "outside_events": 0,
                "first_touch_events": 0, "first_touch_outside_events": 0,
                "lagged_first_touch_outside_events": 0,
                "excluded_prior_mass": 0.0}
    return {**z(), "by_depth": {b: z() for b in DEPTH_BUCKETS},
            "forced_root_bypass_events": 0,
            "forced_root_bypass_outside_events": 0,
            "within_forced_events": 0, "missing_snapshots": 0}


def _pool_counters(acc: Dict[str, Any], snap: Optional[Dict[str, Any]],
                   shape_key: str) -> None:
    """Section 8's online aggregates, pooled across the cohort.

    SUM numerators and denominators; never average per-row rates. A mean of
    means would weigh a 10-event row the same as a 990-event one.
    """
    if snap is None:
        acc["missing_snapshots"] += 1
        return
    block = snap["by_shape"][shape_key]
    for key in ("overall",) + DEPTH_BUCKETS:
        cell = block[key]
        target = acc if key == "overall" else acc["by_depth"][key]
        for f in ("eligible_events", "outside_events", "first_touch_events",
                  "first_touch_outside_events",
                  "lagged_first_touch_outside_events", "excluded_prior_mass"):
            target[f] += cell[f]
    # Forced-root bypasses are reported SEPARATELY and never enter the primary
    # intervention denominator (design section 8).
    acc["forced_root_bypass_events"] += block["forced_root_bypass_events"]
    acc["forced_root_bypass_outside_events"] += block[
        "forced_root_bypass_outside_events"]
    acc["within_forced_events"] += snap["within_forced_events"]


def _finalize_counters(acc: Dict[str, Any]) -> Dict[str, Any]:
    def rates(c):
        c["outside_rate"] = _rate(c["outside_events"], c["eligible_events"])
        c["first_touch_outside_rate"] = _rate(c["first_touch_outside_events"],
                                              c["first_touch_events"])
        c["lagged_first_touch_outside_rate"] = _rate(
            c["lagged_first_touch_outside_events"], c["first_touch_events"])
        # EVENT-WEIGHTED: total mass outside top-K over total eligible events.
        c["mean_excluded_prior_mass"] = _rate(c["excluded_prior_mass"],
                                              c["eligible_events"])
    rates(acc)
    for cell in acc["by_depth"].values():
        rates(cell)
    acc["forced_root_bypass_outside_rate"] = _rate(
        acc["forced_root_bypass_outside_events"],
        acc["forced_root_bypass_events"])
    return acc


def aggregate_shape(rows: Sequence[Dict[str, Any]],
                    shape: Tuple[str, float, float]) -> Dict[str, Any]:
    """Fold per-row results into the rates section 8 gates, AT BOTH INSTANTS.

    Retention runs over the DEDUPLICATED UNION of required edges from both deep
    lines (amendment 4), each evaluated against that instant's own parent-visit
    map. INCONCLUSIVE rows are excluded from the intervention denominator and
    counted separately -- folding them in as either outcome would invent a
    measurement. If a denominator empties, the rate is None and the shape CANNOT
    pass.
    """
    name = shape[0]
    acc = {inst: {"by_role": {r: _pair() for r in DEPTH_NAMES},
                  "by_stratum": {s: _pair() for s in STRATA},
                  "retention_rows": 0,
                  "mis_num": 0, "mis_den": 0, "stab_num": 0, "stab_den": 0,
                  "inconclusive": 0}
           for inst in INSTANTS}
    counters = {inst: _empty_counters() for inst in INSTANTS}
    agreement = {d: {"agree": 0, "disagree": 0,
                     "single_line": 0, "absent_both": 0}
                 for d in DEPTH_NAMES}
    rows_without_stable_reference = 0

    for row in rows:
        snaps = row["snapshots"]
        merged = snaps["reference_lines"]["merged"]
        row_strata = classify_strata(row)
        # Agreement is reported over EVERY row, including unstable ones: a row
        # can be no_stable_reference because of the value gap or the top-two
        # margin while its deep root moves agree, so agreement and stability
        # are different facts and pooling all rows is the honest report.
        for depth_name, a in merged["agreement"].items():
            agreement[depth_name][a["state"]] += 1

        # Section 8's floors concern STABLE deep moves. A row whose deep rungs
        # never agreed has no stable deep move, so it contributes no required
        # edges -- but its selection events still count, because those describe
        # what widening would have done regardless of label.
        stable = row["label"] in STABLE_REFERENCE_LABELS
        if not stable:
            rows_without_stable_reference += 1

        for inst in INSTANTS:
            a = acc[inst]
            visits = (snaps.get("parent_visits") or {}).get(inst) or {}
            if stable:
                a["retention_rows"] += 1
                for edge in merged["required_edges"]:
                    if edge["depth"] >= len(DEPTH_NAMES):
                        continue                 # beyond the two-ply horizon
                    res = edge_retention(edge, visits, shape)
                    # Flat-policy status is recomputed LOCALLY per edge; row
                    # strata apply to every edge of that row.
                    buckets = [a["by_role"][DEPTH_NAMES[edge["depth"]]]]
                    buckets += [a["by_stratum"][s]
                                for s in row_strata | classify_edge_strata(edge)]
                    for bucket in buckets:
                        bucket["required"] += 1
                        bucket["retained"] += 1 if res["retained"] else 0

            iv = intervention_from_snapshots(snaps, name, instant=inst)
            if iv["meaningfully_affected"] is None:
                a["inconclusive"] += 1
            elif row["label"] == "misleading":
                a["mis_den"] += 1
                a["mis_num"] += 1 if iv["meaningfully_affected"] else 0
            elif row["label"] == "stable_negative":
                a["stab_den"] += 1
                a["stab_num"] += 1 if iv["meaningfully_affected"] else 0

            _pool_counters(counters[inst], snaps.get(inst), name)

    def with_rate(p):
        return {**p, "rate": _rate(p["retained"], p["required"])}

    instants: Dict[str, Any] = {}
    for inst in INSTANTS:
        a = acc[inst]
        roles = {r: with_rate(p) for r, p in a["by_role"].items()}
        instants[inst] = {
            "by_role": roles,
            "root_retention": roles["root"]["rate"],
            "depth1_retention": roles["reply"]["rate"],
            "descendant_retention": roles["two_ply"]["rate"],
            "by_stratum": {s: with_rate(p) for s, p in a["by_stratum"].items()},
            # How many rows the retention bars actually rest on.
            "retention_rows": a["retention_rows"],
            "misleading_intervention": _rate(a["mis_num"], a["mis_den"]),
            "stable_intervention": _rate(a["stab_num"], a["stab_den"]),
            "misleading_denominator": a["mis_den"],
            "stable_denominator": a["stab_den"],
            "inconclusive": a["inconclusive"],
        }

    def worst(field):
        """Amendment 4: the floors must pass at BOTH instants, and
        `min(a, b) >= bar` is exactly that. None if either is undefined, since
        an undefined rate is not a satisfied bar."""
        vals = [instants[i][field] for i in INSTANTS]
        return None if any(v is None for v in vals) else min(vals)

    gated = instants[GATING_INSTANT]
    return {
        "shape": name,
        "gated_on": GATING_INSTANT,
        "instants": instants,
        "counters": {i: _finalize_counters(counters[i]) for i in INSTANTS},
        # Reported, never gated (amendment 4). single_line and absent_both are
        # DIFFERENT missingness states and neither enters the denominator.
        "agreement": {d: {**v, "agree_rate": _rate(v["agree"],
                                                   v["agree"] + v["disagree"])}
                      for d, v in agreement.items()},
        # The retention bars rest only on stable-reference-eligible rows; this
        # says how many were set aside, so a floor computed over three rows is
        # not mistaken for one computed over the corpus.
        "rows_without_stable_reference": rows_without_stable_reference,
        "retention_rows": instants[GATING_INSTANT]["retention_rows"],
        # Hoisted for select_shape / validation_verdict: retention is the WORSE
        # instant, intervention is the B=400 number.
        "root_retention": worst("root_retention"),
        "depth1_retention": worst("depth1_retention"),
        "descendant_retention": worst("descendant_retention"),
        "misleading_intervention": gated["misleading_intervention"],
        "stable_intervention": gated["stable_intervention"],
        "misleading_denominator": gated["misleading_denominator"],
        "stable_denominator": gated["stable_denominator"],
        "inconclusive": gated["inconclusive"],
    }


def validation_verdict(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """The frozen three-way precedence (amendment 6a):

        1. FAIL          -- any DEFINED rate misses its bar
        2. INCONCLUSIVE  -- otherwise, any required rate is UNDEFINED
        3. PASS          -- otherwise

    ORDERED, because a result can hold both a defined miss and an undefined rate
    and would otherwise satisfy two verdicts at once. A defined miss is evidence
    and outranks a gap in the evidence; an undefined rate is not a satisfied bar
    and is not a measured miss either.
    """
    failed: List[str] = []
    undefined: List[str] = []
    for rate_name in REQUIRED_RATES:
        value = aggregate.get(rate_name)
        bar, kind = _BARS[rate_name]
        if value is None:
            undefined.append(rate_name)
        elif (value < bar) if kind == "floor" else (value > bar):
            failed.append(rate_name)
    verdict = "FAIL" if failed else ("INCONCLUSIVE" if undefined else "PASS")
    return {"verdict": verdict, "failed": failed, "undefined": undefined}


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
        # Nothing was validated, so there is no validation aggregate to judge.
        # NO_SHAPE_PASSES is a selection outcome, not an INCONCLUSIVE
        # validation, and must not be reported as one.
        return {**chosen, "selected_on": "discovery", "validated": {},
                "validation_verdict": None}
    shape = next(s for s in WIDENING_SHAPES if s[0] == chosen["selected"])
    validated = aggregate_shape(validation, shape)
    return {**chosen, "selected_on": "discovery",
            "discovery": disc,
            "validated": {chosen["selected"]: validated},
            # Amendment 6a: the validation aggregate is JUDGED, not merely
            # computed.
            "validation_verdict": validation_verdict(validated)}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_c.py -v -p no:cacheprovider`
Expected: PASS — 30 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_readout_c.py tests/test_atlas_readout_c.py
git commit -m "feat(atlas-s4): Read-out C edge-union retention at both instants, strata, verdict"
```

---

### Task 6: Artifact schema, provenance and `_jsonable`

**Files:**
- Create: `scripts/GPU/alphazero/atlas_artifact.py`
- Modify: `scripts/GPU/alphazero/build_atlas_corpus.py` — one additive branch in `_jsonable`
- Test: `tests/test_atlas_artifact.py`

> **Why landed Stage 2 code is touched.** The row holds **native Python**, not
> JSON-ready values — tuple-keyed `parent_visits`, and `LegResult` / `BoundaryRecord`
> dataclasses — and `_jsonable` normalizes at the JSON boundary. That is Stage 2's own
> stated principle ("keeps the pure module free of a JSON concern"), and dataclasses are
> the same problem as tuple keys.
>
> Converting the dataclasses earlier, inside `build_row`, is what revision 3 did with
> `vars(l)` — and it broke the seam: Read-out B and `atlas_labelling` read `l.nominal_B`
> by **attribute**, so a row carrying `vars()`-flattened legs cannot be handed to
> `calibrate_gate` at all. The chain hid that by rebuilding a Read-out B row by hand.
>
> The branch is additive and byte-identical for every existing caller, whose payloads
> contain no dataclasses:
>
> ```python
> if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
>     return _jsonable(dataclasses.asdict(obj))
> ```

**Interfaces:**
- Consumes: Stage 3's `snapshots` document verbatim, Task 2/3's feature dicts, and Stage 3's `BoundaryRecord` / `PrefixResult`.
- Produces: `ROW_SCHEMA_VERSION`; `build_row(..., snapshots=...)`; `validate_provenance(prov)`; `emit(run)`.

> **One authoritative `snapshots` document, stored once.** Revision 2 stored the Stage 3
> document under `tracer_snapshots` *and* separately copied `reference_lines` and
> `parent_visits` to the top level — three keys holding overlapping copies of one
> producer's output, none of them the shape Read-out C consumes. The chain then worked
> around it with a hand-built `_c_row`, **recreating exactly the producer/consumer
> surrogate this plan exists to eliminate** and leaving the artifact path itself
> untested.
>
> Now `build_row` takes `snapshots=` and stores it under `snapshots`, unchanged and
> undivided. **An artifact row is therefore already a valid Read-out C row** (it carries
> `snapshots`, `label`, `phase`, `flat_policy`, `near_even`) **and a valid Read-out A row**
> (`label`, `features_at_boundary`, `features_at_400`). The chain drives both read-outs
> from real `build_row` output, so the seam is exercised rather than bypassed.
>
> **Tuple keys become `"a|b"` strings at emission**, via the existing `_jsonable`, and the
> root path `()` becomes the **empty-string key** `""`. That is deterministic, but it is
> surprising enough to pin: a reader who expects `"root"` would silently treat `""` as an
> absent entry. Emission is a report, not a re-loadable input.
>
> **`emit` refuses to serialize a run whose provenance does not validate.** Revision 2
> defined `validate_provenance` and never called it, and its own chain test emitted an
> empty provenance object successfully — a fail-closed check nothing invokes is
> decoration. Digests must also be **hexadecimal**, not merely 40 characters long: a
> 40-character non-hex string is not a SHA-1, and length alone would accept one.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atlas_artifact.py
import json

import pytest

from scripts.GPU.alphazero.atlas_artifact import (
    ROW_SCHEMA_VERSION, build_row, emit, validate_provenance,
)
from scripts.GPU.alphazero.warm_prefix_replay import LegResult

# Valid synthetic provenance. Emission is fail-closed, so serialization tests
# must supply one rather than proving that an empty object gets through.
PROV = {"git_head": "a" * 40, "worktree_clean": True,
        "checkpoint_sha1": "0" * 40}


def _snapshots(**over):
    """The Stage 3 producer document, stored ONCE and undivided."""
    base = {"at_boundary": {"by_shape": {}}, "at_400": {"by_shape": {}},
            "captures": {"at_start": {}, "at_boundary": {}, "at_400": {}},
            "parent_visits": {"at_boundary": {(): 463}, "at_400": {(): 537}},
            "reference_lines": {"at_3200": {"moves": [7]},
                                "at_6400": {"moves": [7]},
                                "merged": {"required_edges": [],
                                           "agreement": {}}}}
    base.update(over)
    return base


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
        snapshots=_snapshots(),
        flat_policy=False, near_even=True)
    base.update(over)
    return base


def test_row_carries_BOTH_feature_captures():
    """Section 6a: B=400 supplies the required 400-tree diagnostic contrast, so
    it must survive into the artifact, not just the boundary capture."""
    r = build_row(**_kw())
    assert r["features_at_boundary"]["one_visit_backup_share"] == 0.4
    assert r["features_at_400"]["one_visit_backup_share"] == 0.3


def test_the_row_stores_ONE_undivided_producer_document():
    """Amendment 4's output is kept whole, under the key Read-out C consumes.
    The tree is gone by the time anything re-reads this row, so a map that is
    not carried is a permanently missing measurement -- and a map carried in
    two overlapping places is how the two copies drift."""
    r = build_row(**_kw())
    assert set(r["snapshots"]["reference_lines"]) == {"at_3200", "at_6400",
                                                      "merged"}
    assert set(r["snapshots"]["parent_visits"]) == {"at_boundary", "at_400"}
    # No duplicated copies and no abolished singular field.
    for gone in ("reference_line", "reference_lines", "parent_visits",
                 "tracer_snapshots"):
        assert gone not in r


def test_an_artifact_row_IS_a_readout_row_for_both_consumers():
    """The seam is the contract: no translation layer exists to drift."""
    r = build_row(**_kw())
    assert {"snapshots", "label", "phase", "flat_policy", "near_even"} <= set(r)
    assert {"label", "features_at_boundary", "features_at_400"} <= set(r)


def test_the_rows_NATIVE_shapes_survive_emission():
    """The row holds native Python and `_jsonable` normalizes at the boundary.

    Tuple KEYS join with "|", so the root path () becomes "" -- deterministic,
    but surprising enough that a reader must not mistake it for an absent
    entry. Dataclasses convert too, which is what lets `legs` stay a list of
    LegResult objects that Read-out B can read by ATTRIBUTE.
    """
    leg = LegResult(nominal_B=400, inherited_I=137, effective=537,
                    root_value=0.25, selected_move=7,
                    selected_move_prior_rank=1, top_share=0.5,
                    top_two_margin=0.2, effective_children=12.0,
                    n_visited_children=20, visit_counts={7: 100})
    r = build_row(**_kw(legs=[leg], snapshots=_snapshots(
        parent_visits={"at_boundary": {(): 463, (7, 3): 12}})))
    back = json.loads(emit({"rows": [r], "provenance": PROV}))["rows"][0]
    assert back["snapshots"]["parent_visits"]["at_boundary"] == {"": 463,
                                                                "7|3": 12}
    assert back["legs"][0]["nominal_B"] == 400
    assert back["legs"][0]["root_value"] == 0.25


def test_row_carries_resets_remaining_strata_and_the_schema_version():
    r = build_row(**_kw())
    assert r["schema_version"] == ROW_SCHEMA_VERSION
    assert r["reset_count"] == 1 and r["reset_rate"] == 0.02
    assert r["last_reset_ply"] == 44 and r["boundary"]["remaining"] == 74
    assert r["near_even"] is True and r["flat_policy"] is False


def test_undefined_values_stay_None_through_emission():
    r = build_row(**_kw(reset_rate=None, last_reset_ply=None, boundary=None,
                        features_at_400=None))
    back = json.loads(emit({"rows": [r], "provenance": PROV}))["rows"][0]
    assert back["reset_rate"] is None and back["last_reset_ply"] is None
    assert back["boundary"] is None and back["features_at_400"] is None


def test_a_row_missing_the_boundary_is_flagged_not_defaulted():
    assert build_row(**_kw(boundary=None))["boundary_missing"] is True


def test_emission_goes_through_jsonable():
    run = {"rows": [], "provenance": PROV,
           "cells": {("discovery", "late", "red"): 12}}
    assert json.loads(emit(run))["cells"] == {"discovery|late|red": 12}


def test_emission_REJECTS_an_unserializable_payload():
    """No default=str: it would stringify a schema defect into a
    plausible-looking value instead of failing. Provenance is VALID here, so
    the TypeError proves the serializer refused -- not the provenance gate."""
    with pytest.raises(TypeError):
        emit({"rows": [{"bad": object()}], "provenance": PROV})


def test_emission_REFUSES_a_run_whose_provenance_does_not_validate():
    """A fail-closed check nothing invokes is decoration. Emission is the one
    place every artifact passes through, so the gate belongs here."""
    for bad in ({}, {"git_head": "a" * 40, "worktree_clean": False,
                     "checkpoint_sha1": "0" * 40}):
        with pytest.raises(ValueError, match="provenance"):
            emit({"rows": [], "provenance": bad})


def test_provenance_fails_closed_on_a_dirty_tree():
    r = validate_provenance({"git_head": "a" * 40, "worktree_clean": False,
                             "checkpoint_sha1": "0" * 40})
    assert r["verdict"] == "PROVENANCE_FAILURE" and "worktree_clean" in r["problems"]


def test_provenance_requires_a_checkpoint_digest():
    r = validate_provenance({"git_head": "a" * 40, "worktree_clean": True,
                             "checkpoint_sha1": ""})
    assert "checkpoint_sha1" in r["problems"]


def test_a_forty_character_non_hexadecimal_digest_is_rejected():
    """Length alone is not a SHA-1. Checking only `len == 40` accepts a
    placeholder, a truncated path, or a typo'd branch name."""
    r = validate_provenance({"git_head": "z" * 40, "worktree_clean": True,
                             "checkpoint_sha1": "not-a-hash" + "x" * 30})
    assert r["verdict"] == "PROVENANCE_FAILURE"
    assert set(r["problems"]) == {"git_head", "checkpoint_sha1"}


def test_valid_provenance_passes():
    r = validate_provenance(PROV)
    assert r["verdict"] == "OK" and r["problems"] == []
    # Upper case is still hexadecimal.
    assert validate_provenance({**PROV, "git_head": "A" * 40})["verdict"] == "OK"
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

The Stage 3 producer document is stored ONCE, undivided, under `snapshots`, so
an artifact row is directly consumable by Read-outs A, B and C with no
translation layer between them to drift. The row holds NATIVE Python -- tuple
keys, LegResult and BoundaryRecord dataclasses -- and `_jsonable` normalizes it
at the JSON boundary, exactly where Stage 2 put that concern.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from .build_atlas_corpus import _jsonable

# 1, not 2: no artifact of this schema has ever been emitted, and a version
# number implying a predecessor invites a reader to hunt for one.
ROW_SCHEMA_VERSION = 1


def build_row(*, game_idx: int, replay_seed: int, target_ply: int, phase: str,
              side: str, split: str, inherited_I: int, reset_count: int,
              reset_rate: Optional[float], last_reset_ply: Optional[int],
              boundary: Optional[Any], legs: Sequence[Any],
              label: str, features_at_boundary: Optional[Dict[str, Any]],
              features_at_400: Optional[Dict[str, Any]],
              snapshots: Dict[str, Any], flat_policy: bool,
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
        # LegResult objects, NOT vars()-flattened dicts: Read-out B and
        # atlas_labelling read `l.nominal_B` by ATTRIBUTE, so a flattened row
        # could not be handed to calibrate_gate at all. `_jsonable` converts
        # them at emission.
        "legs": legs, "label": label,
        # BOTH captures: B=400 supplies section 6's 400-tree diagnostic
        # contrast. Together with `label` this row IS a Read-out A row.
        "features_at_boundary": features_at_boundary,
        "features_at_400": features_at_400,
        # The Stage 3 document, WHOLE and under the key Read-out C consumes:
        # tracer snapshots, captures, both parent-visit maps and both deep
        # lines. Splitting it into overlapping copies is how they drift, and
        # storing it under any other name forces a surrogate row in between.
        "snapshots": snapshots,
        # Strata facts, so Read-outs B and C need no second source.
        "flat_policy": flat_policy, "near_even": near_even,
    }


_SHA1 = re.compile(r"[0-9a-fA-F]{40}\Z")


def validate_provenance(prov: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fails CLOSED. A dirty tree or unidentifiable checkpoint means the run is
    not reconstructible, whatever its numbers say.

    Digests must be HEXADECIMAL, not merely 40 characters: a placeholder, a
    truncated path or a typo'd ref can be 40 characters long and is not a SHA-1.
    """
    prov = prov or {}
    problems = []
    if prov.get("worktree_clean") is not True:
        problems.append("worktree_clean")
    for field in ("checkpoint_sha1", "git_head"):
        value = prov.get(field)
        if not isinstance(value, str) or not _SHA1.match(value):
            problems.append(field)
    return {"verdict": "PROVENANCE_FAILURE" if problems else "OK",
            "problems": problems}


def emit(run: Dict[str, Any]) -> str:
    """Serialize through _jsonable, but ONLY for a run that validates.

    The provenance gate lives here because emission is the one point every
    artifact passes through. A fail-closed check that nothing calls is
    decoration: the previous revision defined `validate_provenance` and never
    invoked it, and its own chain test emitted an empty provenance object.

    Validation runs BEFORE serialization so a payload defect still raises
    TypeError rather than being masked by the gate. NO default=str -- it would
    stringify a schema defect into a plausible-looking value instead of failing.
    """
    checked = validate_provenance(run.get("provenance"))
    if checked["verdict"] != "OK":
        raise ValueError(
            f"refusing to emit: provenance does not validate "
            f"({', '.join(checked['problems'])})")
    return json.dumps(_jsonable(run), indent=2, sort_keys=True)
```

And in `build_atlas_corpus.py`, add `import dataclasses` (the module currently
imports only `argparse`, `json`, `sys` and `pathlib`) plus one additive branch at
the top of `_jsonable`, before the `dict` case:

```python
    # Dataclasses are the same boundary problem as tuple keys: the rows hold
    # LegResult / BoundaryRecord objects because the read-outs address them by
    # attribute, and only the JSON boundary needs them flattened. Additive --
    # every existing caller's payload contains no dataclasses, so their output
    # is byte-identical.
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_atlas_artifact.py -v -p no:cacheprovider`
Expected: PASS — 14 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/atlas_artifact.py \
        scripts/GPU/alphazero/build_atlas_corpus.py \
        tests/test_atlas_artifact.py
git commit -m "feat(atlas-s4): one undivided snapshots document, fail-closed emission"
```

Re-run the Stage 2 CLI suite too — `_jsonable` is shared:
`.venv/bin/python -m pytest tests/test_build_atlas_corpus_cli.py -p no:cacheprovider`
Expected: 13 passed, unchanged.

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
    collect_features, deployability, evaluate_detector, evaluate_detector_both,
)
from scripts.GPU.alphazero.atlas_readout_b import (
    by_stratum_summary, calibrate_gate, natural_convergence_report,
)
from scripts.GPU.alphazero.atlas_readout_c import (
    aggregate_shape, classify_strata, intervention_from_snapshots,
    select_on_discovery_validate_on_selected, validation_verdict,
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
PROV = {"git_head": "a" * 40, "worktree_clean": True,
        "checkpoint_sha1": "0" * 40}


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


@pytest.fixture(scope="module")
def artifact_row(real_row):
    """A REAL `build_row` result.

    Every downstream read-out below consumes THIS, never a hand-written
    stand-in. The producer/consumer seam is the thing under test, and a chain
    test that rebuilds its own row shape tests everything except the seam --
    which is how the previous revision left the artifact path unexercised.
    """
    pre, legs, snaps, obs = (real_row["pre"], real_row["legs"],
                             real_row["snaps"], real_row["obs"])
    caps = snaps["captures"]
    return build_row(
        game_idx=real_row["meta"].game_id, replay_seed=real_row["meta"].seed,
        target_ply=2, phase="opening", side="red", split="discovery",
        inherited_I=pre.inherited_I, reset_count=pre.reset_count,
        reset_rate=pre.reset_rate, last_reset_ply=pre.last_reset_ply,
        # Dataclasses, not vars(): the read-outs address these by attribute and
        # `_jsonable` flattens them at emission.
        boundary=obs.record, legs=legs, label=classify_row(legs),
        features_at_boundary=collect_features(caps["at_start"],
                                              caps["at_boundary"],
                                              obs.record.N_actual),
        features_at_400=collect_features(caps["at_start"], caps["at_400"], 400),
        # SUPPLIED facts in this synthetic chain -- see the Stage 5 handoff note
        # in the completion criteria. Stage 5 must DERIVE them.
        snapshots=snaps, flat_policy=False, near_even=True)


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
    r = evaluate_detector(discovery=rows, validation=rows, replicates=64)
    # Identical features cannot separate; the point is that the REAL path runs
    # and reports a verdict rather than raising. `replicates` is cut from the
    # frozen 10,000 because this row set reaches the bootstrap: 10,000 x 500
    # rank comparisons is several seconds of pure Python for no added signal.
    assert r["verdict"] in {"PASS", "FAIL", "INSUFFICIENT_CLASSES",
                            "INSUFFICIENT_DISCOVERY_CLASSES"}
    d = deployability([real_row["obs"].record.remaining])
    assert d["verdict"] in {"DEPLOYABLE", "NOT_DEPLOYABLE"}


def test_readout_B_consumes_the_ARTIFACT_ROW_directly(artifact_row):
    """A Read-out B row is an artifact row too: `legs`, `phase`, `flat_policy`
    and `near_even` are all already there, and `legs` holds LegResult OBJECTS
    so the attribute access in `_by_b` works on the real row."""
    assert calibrate_gate([artifact_row], "top_share_increase")["verdict"] in {
        "needs review", "no finding"}
    nc = natural_convergence_report([artifact_row])
    assert nc["transition"] == "400->6400" and nc["is_causal_evidence"] is False
    assert "overall" in by_stratum_summary([artifact_row], "top_share_increase")


def test_the_real_ladder_freezes_both_deep_lines_and_both_visit_maps(real_row):
    """The producer half of amendment 4, driven by the real ladder rather than
    by a hand-written stand-in for it."""
    snaps = real_row["snaps"]
    assert set(snaps["reference_lines"]) == {"at_3200", "at_6400", "merged"}
    assert set(snaps["parent_visits"]) == {"at_boundary", "at_400"}
    merged = snaps["reference_lines"]["merged"]
    assert set(merged["agreement"]) == {"root", "reply", "two_ply"}
    # Every edge carries its own parent's priors, and no deep-rung visit count.
    for e in merged["required_edges"]:
        assert e["parent_priors"] and "parent_effective_visits" not in e


def test_readout_C_consumes_the_ARTIFACT_ROW_directly(artifact_row):
    """No surrogate row: `aggregate_shape` is handed `build_row`'s output.

    At active_size=6 the admitted set clamps to n_legal, so these numbers
    cannot be interesting -- the point is that the real seam holds and the
    real path reports rates rather than raising.
    """
    assert isinstance(classify_strata(artifact_row), set)
    for instant in ("at_boundary", "at_400"):
        iv = intervention_from_snapshots(artifact_row["snapshots"], "c4a05",
                                         instant=instant)
        assert iv["verdict"] in {"OK", "INCONCLUSIVE", "NO_EVENTS",
                                 "NO_SNAPSHOT"}
    agg = aggregate_shape([artifact_row], SHAPE)
    assert agg["gated_on"] == "at_400"
    assert set(agg["instants"]) == {"at_boundary", "at_400"}
    assert set(agg) >= {"root_retention", "misleading_intervention",
                        "inconclusive", "counters", "agreement",
                        "retention_rows", "rows_without_stable_reference"}
    assert validation_verdict(agg)["verdict"] in {"FAIL", "INCONCLUSIVE", "PASS"}


def test_readout_C_selection_and_verdict_run_on_real_artifact_rows(artifact_row):
    """Labels are forced, because whatever the FakeEvaluator ladder happens to
    classify this position as must not decide whether the test exercises the
    intervention denominators."""
    rows = [{**artifact_row, "label": "misleading"},
            {**artifact_row, "label": "stable_negative"}]
    r = select_on_discovery_validate_on_selected(rows, rows)
    assert r["selected_on"] == "discovery"
    assert set(r["validated"]) <= {r["selected"]}
    if r["selected"] is None:
        assert r["validation_verdict"] is None
    else:
        assert r["validation_verdict"]["verdict"] in {"FAIL", "INCONCLUSIVE",
                                                      "PASS"}


def test_readout_A_dual_pipeline_consumes_the_artifact_row(artifact_row):
    """The detector row IS the artifact row -- no translation layer to drift."""
    rows = ([{**artifact_row, "label": "misleading"} for _ in range(20)]
            + [{**artifact_row, "label": "stable_negative"} for _ in range(25)])
    r = evaluate_detector_both(rows, rows, replicates=32)
    assert r["authoritative"] == "features_at_boundary"
    assert r["verdict"] in {"PASS", "FAIL", "LATE_ONLY_SEPARATION",
                            "INSUFFICIENT_CLASSES",
                            "INSUFFICIENT_DISCOVERY_CLASSES"}
    assert r["row_overlap"]["discovery"]["identical"] is True
    assert r["row_overlap"]["validation"]["identical"] is True


def test_a_real_row_survives_the_artifact_boundary(artifact_row, real_row):
    back = json.loads(emit({"rows": [artifact_row],
                            "provenance": PROV}))["rows"][0]
    assert back["inherited_I"] == real_row["pre"].inherited_I
    assert len(back["legs"]) == 4
    assert back["features_at_boundary"] is not None
    assert back["features_at_400"] is not None
    # Amendment 4's producer output survives the JSON boundary, tuple keys and
    # all: the root path () emits as the empty-string key.
    assert set(back["snapshots"]["reference_lines"]) == {"at_3200", "at_6400",
                                                         "merged"}
    assert "" in back["snapshots"]["parent_visits"]["at_400"]


def test_emission_of_a_real_run_still_fails_closed_on_provenance(artifact_row):
    """The gate is not bypassed by a row that is otherwise perfectly valid."""
    with pytest.raises(ValueError, match="provenance"):
        emit({"rows": [artifact_row], "provenance": {}})
```

- [ ] **Step 2: Run, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_atlas_readout_chain.py -v -p no:cacheprovider`
Expected: PASS — 11 passed.

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
- [ ] **Amendment 4's producer contract.** Both deep lines captured while their states exist, selected by **leg index** not `running_B`; complete depth-two parent-visit maps at the boundary and at `B = 400`; the union deduplicated by `(parent path, move)` with both replies retained on disagreement; agreement judged on the complete edge with single-line depths outside the denominator; parent priors **asserted** equal where both rungs captured the same parent; no deep-rung visit count emitted at all.
- [ ] §5 labelling exact: stable reference needs all three conditions; misleading is an OR; stable-negative an AND; ambiguous kept and counted; value and move components reported separately.
- [ ] §3 sizing matches the frozen formula and **fails closed** on a zero class frequency or a requirement above 400; the final capacity gate needs ≥20 / ≥25.
- [ ] Read-out A collects **exactly five** frozen features; standardization is learned on **discovery only**; AUC / bootstrap-lower-bound / flag-rate / precision bars all enforced; `INSUFFICIENT_CLASSES` fails closed.
- [ ] **Read-out A runs the identical pipeline on both feature sets**, boundary authoritative; `LATE_ONLY_SEPARATION` requires boundary `FAIL` **and** `B = 400` `PASS`, both exactly, **and zero boundary missing-feature rejections in either split**; a boundary insufficiency or a blocked lateness reading is reported **as itself**, never promoted to a finding about timing; row overlap is reported for **discovery and validation separately**, and not gated.
- [ ] Deployability: `remaining == 0` non-actionable, **median zero fails**, strata **reported not gated**, empty set yields `None`.
- [ ] Read-out B computes metrics at **all four rungs**; `closes_half` guards a zero gap and checks **both** deep rungs for the **same** metric; persistence is joint; the denominator is `eligible_triggers` with the **base-rate margin**; the verdict is *needs review*, never *invalid*.
- [ ] **Read-out C aggregates over the deduplicated edge union at both instants**: `K(n)` reads the instant's own parent-visit map (absent path ⇒ `0` ⇒ `K(0) = 1`); root and reply floors must pass at **both**; the `B = 400` intervention drives the bars and the boundary one is reported; intervention must **also** pass under `K(n+14)`, otherwise **inconclusive**; depth buckets, event-weighted excluded prior mass, forced-root bypasses, forced-simulation counts and per-stratum retention are all reported; `classify_edge_strata` is **wired into the aggregation**, not merely defined; lexicographic selection with a named `NO_SHAPE_PASSES`.
- [ ] **Only stable-reference-eligible rows feed the retention bars**, via an allow-list; selection-event counters still cover every row; `retention_rows` and `rows_without_stable_reference` are reported so a floor resting on three rows is not mistaken for one resting on the corpus.
- [ ] **Agreement carries four states** — `agree` / `disagree` / `single_line` / `absent_both` — with the last two counted separately and neither in the denominator.
- [ ] **The selected shape's validation aggregate receives the three-way verdict** with the frozen precedence `FAIL > INCONCLUSIVE > PASS`, and `NO_SHAPE_PASSES` yields **no** verdict rather than an `INCONCLUSIVE` one.
- [ ] Artifact stores the Stage 3 producer document **once, undivided, under `snapshots`**, so an artifact row is directly consumable by Read-outs A and C; it carries resets, `remaining`, `boundary_missing` and `None` for every undefined value through emission; `_jsonable` at the boundary with the `()` → `""` root key pinned.
- [ ] **`emit` refuses a run whose provenance does not validate**, checking hexadecimal digests rather than length alone, and validating *before* serializing so a payload defect still raises `TypeError`.
- [ ] Real Stage 3 ladder output drives **all three** read-outs **through a real `build_row` result** — no hand-written row stands in for the artifact anywhere in the chain.

### Handoff to Stage 5 — `flat_policy` and `near_even` are SUPPLIED, not derived

Stage 4 accepts both as caller-supplied booleans and every test hardcodes them. That is
correct for a synthetic stage — they are inputs to strata classification, not something
Stage 4 measures — but it means **the one seam Stage 4 cannot qualify is the one that
computes them.** Read-out B's `late` / `flat_policy` / `near_even` strata and Read-out C's
`root_flat` / `near_even` strata are only as good as that unqualified producer.

Stage 5 must derive both from **already-frozen measured fields** and qualify the
producer → `build_row` seam the same way this stage qualifies the others:

| field | frozen definition | available from |
|---|---|---|
| `flat_policy` | normalized policy entropy ≥ `0.90` **and** top prior ≤ `0.025` (§8, "use the existing flat-policy definition ... rather than inventing strata after measurement") | `captures[*]["policy_entropy"]` and the root priors — the same predicate `atlas_readout_c._is_flat` already applies per edge |
| `near_even` | `\|V_stm\| ≤ 0.30` (§8) | `LegResult.root_value` at nominal `B = 400` |
| `phase` | ply bounds 0–30 / 31–60 / 61–90 / 91+ (§3) | `corpus_geometry.phase_for_ply(target_ply)` |

Note `_is_flat` already exists and is applied to reference-line parents; the root-level
`flat_policy` must use **that same predicate**, not a second implementation of it.
Deriving them in Stage 4 would mean writing a producer no Stage 4 test can drive with
real data — which is how the atlas acquired its first three phantom names.

### Test counting — recounted, and deliberately not frozen

The revision-1 target ("78 new tests, full suite 2513") was **stale and wrong in both
halves**: it predated this rewrite, and its per-file figures were already inaccurate for
Read-out B (12 claimed, 14 present) and Read-out C (14 claimed, 16 present). It is
removed rather than adjusted.

Counted from the test functions actually written in this plan:

| file | tests |
|---|---:|
| `tests/test_atlas_producer_closure.py` | 13 |
| `tests/test_atlas_labelling.py` | 10 |
| `tests/test_atlas_readout_a.py` | 27 |
| `tests/test_atlas_readout_b.py` | 14 |
| `tests/test_atlas_readout_c.py` | 30 |
| `tests/test_atlas_artifact.py` | 14 |
| `tests/test_atlas_readout_chain.py` | 11 |
| **planned new** | **119** |

Stage 3 qualified at **2435**, so the expected full-suite total is **2554**.

**This is a derivation to re-verify, not an acceptance number.** At qualification:

1. Recount `def test_` in the files as they were actually written — the plan is a
   starting point and the count moves when implementation reveals a case worth pinning.
2. The full-suite delta must equal that recount. **Any other delta means a pre-existing
   test changed behaviour**, which must be explained before Stage 4 is called qualified —
   that is the whole value of the number, and it is why the arithmetic is stated here
   rather than left implicit.
3. Read the exit code from the process: `... > /tmp/s4.out 2>&1; echo "REAL_EXIT=$?"`.
   **`cmd | tail` reports the pipe's exit code** and has masked a collection error as
   `exit 0` twice in this line of work.

## Out of scope

No reservoir generation, no checkpoint loading, no MLX execution, no measurement run.
The three distribution gaps — real-scale throughput, the `remaining` distribution, and
the inheritance-reset rate — remain **operator/pilot measurements**. Stage 5 is planned
only after these interfaces exist and qualify.
